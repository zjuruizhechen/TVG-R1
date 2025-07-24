# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import warnings

import nncore
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import ModuleList, PositionalEncoding, Sequential, TransformerEncoderLayer, xavier_init_
from nncore.ops import temporal_iou
from transformers import AutoConfig, AutoModel, Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLModel
from transformers.activations import ACT2CLS, ACT2FN
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from .blocks import ConvHead, ConvPyramid, LearnableEmbedding, Scale
from .generator import PointGenerator
from .loss import BundleLoss


def cache_state_hook(module, args):
    module.state = args[0]


class AgentQwen2VLConfig(Qwen2VLConfig):
    model_type = 'agent_qwen2_vl'


class AgentQwen2VisionTransformerPretrainedModel(Qwen2VisionTransformerPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = False

    # add support for gradient checkpointing
    # https://github.com/huggingface/transformers/pull/34724
    def forward(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(blk.__call__, hidden_states, cu_seqlens,
                                                                  rotary_pos_emb)
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


class AgentQwen2VLModel(Qwen2VLModel):
    config_class = AgentQwen2VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.norm.register_forward_pre_hook(cache_state_hook)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        # ensure gradient tracking (in case that embed_tokens has been frozen)
        assert input_ids is None and inputs_embeds is not None
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad = True
        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)


class AgentQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    config_class = AgentQwen2VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual = AgentQwen2VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = AgentQwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None

        if self.config.role in ('all_in_one', 'grounder'):
            hidden_size, hidden_act = self.config.hidden_size, self.config.hidden_act

            self.dims = 256

            self.vis_proj = Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.dims))
            self.reg_proj = Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, self.dims))
            self.vis_norm = nn.LayerNorm(self.dims)
            self.vis_fuse = ModuleList(
                TransformerEncoderLayer(self.dims, act_cfg=ACT2FN[hidden_act]),
                TransformerEncoderLayer(self.dims, act_cfg=ACT2FN[hidden_act]),
                TransformerEncoderLayer(self.dims, act_cfg=ACT2FN[hidden_act]))

            self.vis_pos = PositionalEncoding(self.dims, normalize=True, learnable=False)
            self.vis_emb = LearnableEmbedding(self.dims)
            self.reg_emb = LearnableEmbedding(self.dims)

            self.strides = (1, 2, 4, 8)
            self.vis_pad_length = self.strides[-1]

            self.pyramid = ConvPyramid(self.dims, self.strides, act_cls=ACT2CLS[hidden_act])
            self.class_head = ConvHead(self.dims, 1, act_cls=ACT2CLS[hidden_act])
            self.coord_head = ConvHead(self.dims, 2, act_cls=ACT2CLS[hidden_act])

            self.generator = PointGenerator(self.strides, 1024)
            self.coef = Scale(self.strides)
            self.bundle_loss = BundleLoss(
                sample_radius=1.5,
                loss_cls=dict(type='FocalLoss', reduction='none', loss_weight=5.0),
                loss_reg=dict(type='L1Loss', reduction='none', loss_weight=1.0),
                loss_sal=dict(type='SampledNCELoss', direction='row', loss_weight=0.05))

        self.post_init()

    def reset_conv_parameters(self):
        for s in ('pyramid', 'class_head', 'coord_head'):
            b = getattr(self, s, None)
            if b is None:
                continue
            for n, m in b.named_modules():
                if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                    print(f'Reset parameters of {b.__class__.__name__} {n} ({m.__class__.__name__})')
                    xavier_init_(m, distribution='uniform')

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                rope_deltas=None,
                timestamps=None,
                saliency=None,
                pos_clip=None):
        mode = 'training' if self.training else 'caching' if (
            past_key_values is None or len(past_key_values) == 0) else 'generating'

        # https://github.com/huggingface/transformers/pull/33487
        if position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

        if mode in ('training', 'caching'):
            vision_s_inds = torch.nonzero(input_ids == self.config.vision_start_token_id).tolist()
            vision_e_inds = torch.nonzero(input_ids == self.config.vision_end_token_id).tolist()
            assert len(vision_s_inds) == len(vision_e_inds)

            self.cache_vision_inds = [[] for _ in range(input_ids.size(0))]
            for i in range(len(vision_s_inds)):
                assert vision_s_inds[i][0] == vision_e_inds[i][0]
                self.cache_vision_inds[vision_s_inds[i][0]].append([vision_s_inds[i][1] + 1, vision_e_inds[i][1]])

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=not self.training,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas)

        if mode == 'caching':
            self.cache_norm_state = self.model.norm.state
            self.reg = []
            self.sal = []

        if mode == 'training' and timestamps is not None:
            loss_regs, avg_factors = [], []
            shift_labels = labels[..., 1:].contiguous()
            for batch_idx, (vision_inds, ts) in enumerate(zip(self.cache_vision_inds, timestamps)):
                # only consider the first video
                s, e = vision_inds[0]

                # spatial merge size set to 2
                window = int(video_grid_thw[0][1] * video_grid_thw[0][2] / 4)
                assert video_grid_thw[0][0] * window == e - s

                inds = torch.where(shift_labels[batch_idx] == self.config.reg_token_id)[0]
                reg_tokens = self.reg_proj(self.model.norm.state[batch_idx, inds, None])
                # reg_tokens: num_reg_tokens * 1 * channel

                vis_tokens = self.model.norm.state[batch_idx, None, s:e]
                vis_tokens = vis_tokens.transpose(-1, -2)
                vis_tokens = F.avg_pool1d(vis_tokens.float(), window, stride=window).to(vis_tokens.dtype)
                vis_tokens = vis_tokens.transpose(-1, -2)
                vis_tokens = self.vis_proj(vis_tokens).repeat(reg_tokens.size(0), 1, 1)
                # vis_tokens: num_reg_tokens * num_frames * channel

                vis_tokens = self.vis_emb(vis_tokens)
                reg_tokens = self.reg_emb(reg_tokens)
                pe = self.vis_pos(vis_tokens).to(vis_tokens.dtype)

                joint_tokens = torch.cat((vis_tokens + pe, reg_tokens), dim=1)
                collected = [joint_tokens]
                for blk in self.vis_fuse:
                    collected.append(blk(collected[-1]))
                collected = collected[1:]
                joint_tokens = torch.cat(collected)
                joint_tokens = self.vis_norm(joint_tokens)

                video_emb = joint_tokens[:, :-1]
                # video_emb: num_reg_tokens * num_frames * channel

                query_emb = joint_tokens[:, -1:]
                # query_emb: num_reg_tokens * 1 * channel

                b, t, c = video_emb.size()
                video_msk = video_emb.new_ones(b, t)

                if t < self.vis_pad_length:
                    emb_pad = video_emb.new_zeros(b, self.vis_pad_length - t, c)
                    msk_pad = video_msk.new_zeros(b, self.vis_pad_length - t)
                    pymid_emb = torch.cat((video_emb, emb_pad), dim=1)
                    pymid_msk = torch.cat((video_msk, msk_pad), dim=1)
                else:
                    pymid_emb, pymid_msk = video_emb, video_msk

                pymid, pymid_msk = self.pyramid(pymid_emb, pymid_msk, return_mask=True)
                if not len(pymid) == len(pymid_msk) == len(self.strides):
                    warnings.warn(f'pyramid size mismatch: {len(pymid)} {len(pymid_msk)} {len(self.strides)}')

                point = self.generator(pymid)

                out_class = [self.class_head(e) for e in pymid]
                out_class = torch.cat(out_class, dim=1)

                out_coord = [self.coef(self.coord_head(e).exp(), i) for i, e in enumerate(pymid)]
                out_coord = torch.cat(out_coord, dim=1)

                data = dict(
                    point=point,
                    video_emb=video_emb,
                    query_emb=query_emb,
                    video_msk=video_msk,
                    pymid_msk=pymid_msk,
                    out_class=out_class,
                    out_coord=out_coord,
                    boundary=point.new_tensor(ts),
                    saliency=saliency[batch_idx].unsqueeze(0),
                    pos_clip=pos_clip[batch_idx].unsqueeze(0))

                losses = self.bundle_loss(data, dict())
                # print({k: v.item() for k, v in losses.items()})

                loss_regs.append(sum(v for v in losses.values()))
                avg_factors.append(len(ts))

            assert len(loss_regs) in (1, 2) and len(loss_regs) == len(avg_factors)

            if len(loss_regs) == 2 and loss_regs[0] > loss_regs[1]:
                loss_reg, avg_factor = loss_regs[1], avg_factors[1]
            else:
                loss_reg, avg_factor = loss_regs[0], avg_factors[0]

            if avg_factor > 0:
                outputs.loss = outputs.loss + loss_reg / avg_factor
        elif mode == 'generating':
            logits = outputs.logits[0, -1]
            if logits.argmax() == self.config.reg_token_id:
                assert self.model.norm.state.size() == (1, 1, self.config.hidden_size)

                # only consider the first video
                s, e = self.cache_vision_inds[0][0]

                # spatial merge size set to 2
                window = int(video_grid_thw[0][1] * video_grid_thw[0][2] / 4)
                assert video_grid_thw[0][0] * window == e - s

                reg_tokens = self.reg_proj(self.model.norm.state)
                # reg_tokens: num_reg_tokens * 1 * channel

                vis_tokens = self.cache_norm_state[:, s:e]
                vis_tokens = vis_tokens.transpose(-1, -2)
                vis_tokens = F.avg_pool1d(vis_tokens.float(), window, stride=window).to(vis_tokens.dtype)
                vis_tokens = vis_tokens.transpose(-1, -2)
                vis_tokens = self.vis_proj(vis_tokens).repeat(reg_tokens.size(0), 1, 1)
                # vis_tokens: num_reg_tokens * num_frames * channel

                vis_tokens = self.vis_emb(vis_tokens)
                reg_tokens = self.reg_emb(reg_tokens)
                pe = self.vis_pos(vis_tokens).to(vis_tokens.dtype)

                joint_tokens = torch.cat((vis_tokens + pe, reg_tokens), dim=1)
                for blk in self.vis_fuse:
                    joint_tokens = blk(joint_tokens)
                joint_tokens = self.vis_norm(joint_tokens)

                video_emb = joint_tokens[:, :-1]
                # video_emb: num_reg_tokens * num_frames * channel

                query_emb = joint_tokens[:, -1:]
                # query_emb: num_reg_tokens * 1 * channel

                b, t, _ = video_emb.size()
                video_msk = video_emb.new_ones(b, t)

                pymid = self.pyramid(video_emb, video_msk)
                point = self.generator(pymid)

                out_class = [self.class_head(e).sigmoid() for e in pymid]
                out_class = torch.cat(out_class, dim=1)

                out_coord = [self.coef(self.coord_head(e).exp(), i) for i, e in enumerate(pymid)]
                out_coord = torch.cat(out_coord, dim=1)

                sal = out_class[0]
                bnd = out_coord[0]

                bnd[:, 0] *= -1
                bnd *= point[:, 3, None].repeat(1, 2)
                bnd += point[:, 0, None].repeat(1, 2)
                bnd /= t
                bnd = torch.cat((bnd, sal), dim=-1)

                _, inds = bnd[:, -1].sort(descending=True)
                bnd = bnd[inds]

                # hard coding nms config here
                nms_cfg = dict(type='normal', thres=0.75)
                assert nms_cfg['type'] in ('normal', 'linear', 'gaussian')

                for i in range(bnd.size(0)):
                    max_idx = bnd[i:, -1].argmax(dim=0)
                    bnd = nncore.swap_element(bnd, i, max_idx + i)
                    iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

                    if nms_cfg['type'] == 'normal':
                        bnd[i + 1:, -1][iou >= nms_cfg['thres']] = 0
                    elif nms_cfg['type'] == 'linear':
                        bnd[i + 1:, -1] *= 1 - iou
                    else:
                        bnd[i + 1:, -1] *= (-iou.pow(2) / nms_cfg['sigma']).exp()

                # save top-100 predictions
                self.reg.append(bnd[:100])

                # save all saliency scores
                self.sal.append(sal)

        return outputs


# set the patched model to a vision model
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES[AgentQwen2VLConfig.model_type] = 'AgentQwen2VLForConditionalGeneration'

AutoConfig.register(AgentQwen2VLConfig.model_type, AgentQwen2VLConfig)
AutoModel.register(AgentQwen2VLConfig, AgentQwen2VLForConditionalGeneration)
