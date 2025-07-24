# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy

from torch.utils.data import Dataset

from videomind.constants import GROUNDER_PROMPT, REG_TOKEN


class GroundingDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args):
        super(GroundingDataset, self).__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = len(anno['query'].split(' '))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if data_args.min_video_len >= 0 and anno.get('duration', float('inf')) < data_args.min_video_len:
                continue
            if data_args.max_video_len >= 0 and anno.get('duration', 0) > data_args.max_video_len:
                continue
            annos.append(anno)

        self.annos = annos
        self.raw_length = len(raw_annos)
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])

        video_path, duration, query, span = anno['video_path'], anno['duration'], anno['query'], anno['span']

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 150,
                'fps': 1.0
            }, {
                'type': 'text',
                'text': GROUNDER_PROMPT.format(query)
            }]
        }, {
            'role': 'assistant',
            'content': f'The relevant moment happens in {REG_TOKEN}.'
        }]

        meta = dict(messages=messages, span=span, duration=duration)
        return meta
