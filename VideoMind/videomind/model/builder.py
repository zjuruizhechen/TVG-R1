# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import warnings

import nncore
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel, AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration


def get_auto_device(device):
    try:
        import torch_npu
        has_npu = torch_npu.npu.is_available()
    except ImportError:
        has_npu = False

    return 'cuda' if torch.cuda.is_available() else 'npu' if has_npu else 'cpu'


def build_model(model_path, config=None, is_trainable=False, merge_adapter=False, device='auto', dtype=torch.float16):
    # set do_resize to false to avoid duplicated resizing
    # https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
    processor = AutoProcessor.from_pretrained(model_path, do_resize=False)

    # eager attention has known & unknown bugs
    # [4.46.2] broken causality fp16: https://github.com/huggingface/transformers/issues/35151
    # [4.48.1] broken sliding window: https://github.com/huggingface/transformers/issues/35924
    attn_implementation = 'sdpa'

    config = config or AutoConfig.from_pretrained(model_path)

    adapter_path = nncore.join(model_path, getattr(config, 'role', 'unknown'))
    partial_path = nncore.join(model_path, 'pytorch_model.safetensors')

    if nncore.is_dir(adapter_path) or nncore.is_file(partial_path):
        print(f'Loading base model from {config.base_model_path}...')
        model = AutoModel.from_pretrained(
            config.base_model_path,
            config=config,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype)

        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path)
        except OSError:
            warnings.warn('generation_config.json not found')

        meta_state_dict = {
            n: torch.empty_like(p, device='cpu')
            for n, p in model.named_parameters() if p.device == torch.device('meta')
        }
        model.load_state_dict(meta_state_dict, strict=False, assign=True)

        size = (model.model.embed_tokens.num_embeddings, model.model.embed_tokens.embedding_dim)
        if model.model.embed_tokens.weight.size() != size:
            print(f'Resizing embed_tokens to {size}...')
            model.model.embed_tokens.weight = nn.Parameter(model.model.embed_tokens.weight.new_empty(size))

        size = (model.lm_head.out_features, model.lm_head.in_features)
        if model.lm_head.weight.size() != size:
            print(f'Resizing lm_head to {size}...')
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.new_empty(size))

        if nncore.is_dir(adapter_path):
            print(f'Loading adapter from {adapter_path}...')
            # transformers integration does not support merge_and_unload, use peft instead
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                adapter_name=config.role,
                is_trainable=is_trainable,
                low_cpu_mem_usage=True,
                torch_device=str(model.device))

        if nncore.is_file(partial_path):
            print(f'Loading state dict from {partial_path}...')
            _, unexpected = load_model(model, partial_path, strict=False, device=str(model.device))
            assert len(unexpected) == 0, f'unexpected parameters: {unexpected}'

        if merge_adapter and nncore.is_dir(adapter_path):
            print('Merging adapter and unloading...')
            model = model.merge_and_unload()
            model._hf_peft_config_loaded = False
    else:
        print(f'Loading full model from {model_path}...')

        if len(config.architectures) == 1 and config.model_type == 'qwen2_vl':
            model_cls = Qwen2VLForConditionalGeneration
        else:
            model_cls = AutoModel

        model = model_cls.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype)

    if not is_trainable:
        device = get_auto_device(device) if device == 'auto' else device
        model = model.to(device).eval()

    return model, processor
