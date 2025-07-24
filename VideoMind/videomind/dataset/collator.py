# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import warnings

import torch
from torch.nn.utils.rnn import pad_sequence

from videomind.constants import IGNORE_INDEX


class HybridDataCollator(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [d['input_ids'] for d in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = [d['labels'] for d in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        assert input_ids.size() == labels.size()

        seq_len, max_len = input_ids.size(1), self.tokenizer.model_max_length
        if seq_len > max_len:
            warnings.warn(f'The length of input sequence is exceeding model max length: {seq_len} > {max_len}')
            input_ids, labels = input_ids[:, :max_len], labels[:, :max_len]

        data = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids != self.tokenizer.pad_token_id)

        for key in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'):
            if key in batch[0]:
                data[key] = torch.cat([d[key] for d in batch])

        for key in ('timestamps', 'saliency', 'pos_clip'):
            if key in batch[0]:
                data[key] = [d[key] for d in batch]

        return data
