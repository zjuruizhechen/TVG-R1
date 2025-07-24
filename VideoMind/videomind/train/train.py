# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from dataclasses import dataclass, field
from typing import Optional

import nncore
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments

from videomind.constants import REG_TOKEN, SEG_E_TOKEN, SEG_S_TOKEN
from videomind.dataset import HybridDataCollator, HybridDataset
from videomind.model import MODELS
from videomind.model.builder import build_model
from videomind.train.custom_trainer import CustomTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    conv_type: Optional[str] = field(default=None)
    role: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    datasets: Optional[str] = field(default=None)
    min_video_len: Optional[int] = field(default=-1)
    max_video_len: Optional[int] = field(default=-1)
    min_num_words: Optional[int] = field(default=-1)
    max_num_words: Optional[int] = field(default=-1)
    max_retries: Optional[int] = field(default=10)


@dataclass
class CustomArguments:
    optim: Optional[str] = field(default='adamw_torch')
    group_by_data_type: Optional[bool] = field(default=True)
    merge_adapter: Optional[bool] = field(default=False)
    lora_enable: Optional[bool] = field(default=False)
    lora_type: Optional[str] = field(default='qkvo')
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_bias: Optional[str] = field(default='none')
    lora_lr: Optional[float] = field(default=None)
    head_lr: Optional[float] = field(default=None)
    tuning_modules: Optional[str] = field(default=None)
    save_full_model: Optional[bool] = field(default=False)
    remove_unused_columns: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(CustomArguments, TrainingArguments):
    pass


def get_target_modules(model, lora_type, base_model):
    lora_type = lora_type.split('_')
    assert all(t in ('qkvo', 'linear', 'all') for t in lora_type)

    if base_model == 'qwen2_vl':
        # all qkvo layers in the visual encoder and the llm
        qkvo_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn.qkv', 'attn.proj']

        target_modules = set()
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if 'all' not in lora_type and 'visual' in n:
                continue
            if 'qkvo' in lora_type and not any(n.endswith(k) for k in qkvo_keys):
                continue
            target_modules.add(n)
    else:
        raise ValueError(f'unknown base model: {base_model}')

    return target_modules


def train(TrainingArguments, Trainer):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert model_args.role in ('all_in_one', 'planner', 'grounder', 'verifier', 'answerer')

    config_cls, model_cls = MODELS[model_args.base_model]

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    config = config_cls.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype)
    config.update(model_args.__dict__)

    if config.model_type == 'agent_qwen2_vl':
        model, processor = build_model(
            model_args.model_name_or_path,
            config=config,
            is_trainable=True,
            merge_adapter=training_args.merge_adapter,
            dtype=dtype)
    else:
        # set do_resize to false to avoid duplicated resizing
        # https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, do_resize=False)

        # eager attention has known & unknown bugs
        # [4.46.2] broken causality fp16: https://github.com/huggingface/transformers/issues/35151
        # [4.48.1] broken sliding window: https://github.com/huggingface/transformers/issues/35924
        model = model_cls.from_pretrained(model_args.model_name_or_path, config=config, attn_implementation='sdpa')

        # save base model path for inference
        model.config.base_model_path = model_args.model_name_or_path

        # conv parameters may become inf after casting to fp16
        model.reset_conv_parameters()

        model.requires_grad_(False)

    if training_args.lora_enable and not isinstance(model, PeftModel):
        target_modules = get_target_modules(model, training_args.lora_type, model.config.base_model)
        tune_lm_head = model.config.role in ('all_in_one', 'grounder', 'verifier')
        print(f'LoRA target modules: {target_modules}')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=target_modules,
            modules_to_save=['embed_tokens', 'lm_head'] if tune_lm_head else None)
        # transformers integration does not support merge_and_unload, use peft instead
        model = get_peft_model(model, lora_config, adapter_name=model_args.role)

    new_tokens = processor.tokenizer.add_special_tokens(
        dict(additional_special_tokens=[REG_TOKEN, SEG_S_TOKEN, SEG_E_TOKEN]))
    print(f'Added {new_tokens} new token(s)')

    model.config.reg_token_id = processor.tokenizer.convert_tokens_to_ids(REG_TOKEN)
    model.config.seg_s_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_S_TOKEN)
    model.config.seg_e_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_E_TOKEN)

    if new_tokens > 0 and len(processor.tokenizer) > model.config.vocab_size:
        print(f'Expanding vocab size: {model.config.vocab_size} -> {len(processor.tokenizer)}')
        model.resize_token_embeddings(len(processor.tokenizer))
        i_emb = model.get_input_embeddings().weight.data
        o_emb = model.get_output_embeddings().weight.data
        i_emb[-new_tokens:] = i_emb[:-new_tokens].mean(0, keepdim=True)
        o_emb[-new_tokens:] = o_emb[:-new_tokens].mean(0, keepdim=True)

    tuning_modules = [] if training_args.tuning_modules is None else training_args.tuning_modules.split(',')

    head_keys = [
        'vis_proj', 'reg_proj', 'vis_fuse', 'vis_norm', 'vis_pos', 'vis_emb', 'reg_emb', 'pyramid', 'class_head',
        'coord_head', 'coef', 'bundle_loss'
    ]

    for n, p in model.named_parameters():
        # embed_tokens and lm_head might be handled by lora
        if not training_args.lora_enable and new_tokens > 0 and any(k in n for k in ('embed_tokens', 'lm_head')):
            p.requires_grad = True

        if 'projector' in tuning_modules and 'visual.merger' in n:
            p.requires_grad = True

        if model_args.role in ('all_in_one', 'grounder') and any(k in n for k in head_keys):
            p.requires_grad = True

    if training_args.local_rank in (0, -1):
        for n, p in model.named_parameters():
            print(p.requires_grad, p.dtype, p.shape, n)

        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = round(learnable_params / total_params * 100, 2) if total_params > 0 else 0
        print(f'Total params: {total_params} Learnable params: {learnable_params} ({ratio}%)')

        i_size = model.get_input_embeddings().num_embeddings
        o_size = model.get_output_embeddings().out_features
        assert i_size == o_size, (i_size, o_size)
        print(f'Tokenizer size: {len(processor.tokenizer)} Vocab size: {model.config.vocab_size} Embed size: {i_size}')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=HybridDataCollator(processor.tokenizer),
        train_dataset=HybridDataset(processor, model.config, model_args, data_args, training_args),
        processor=processor,
        head_keys=head_keys)

    has_ckpt = bool(nncore.find(training_args.output_dir, 'checkpoint-*'))
    trainer.train(resume_from_checkpoint=has_ckpt)

    trainer.save_state()
    trainer.gather_and_save_model()


if __name__ == '__main__':
    train(TrainingArguments, CustomTrainer)
