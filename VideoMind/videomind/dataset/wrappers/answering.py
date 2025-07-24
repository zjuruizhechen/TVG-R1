# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy
import random

from torch.utils.data import Dataset

from videomind.utils.parser import parse_span


class AnsweringDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args):
        super(AnsweringDataset, self).__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = len(anno['question'].split(' ')) + len(anno['answer'].split(' '))
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

        video_path, question, answer = anno['video_path'], anno['question'], anno['answer']

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28,
                'max_frames': 32,
                'fps': 2.0
            }, {
                'type': 'text',
                'text': question
            }]
        }, {
            'role': 'assistant',
            'content': answer
        }]

        meta = dict(messages=messages)
        return meta


class AnsweringCropDataset(AnsweringDataset):

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])

        video_path, question, answer = anno['video_path'], anno['question'], anno['answer']

        if anno.get('no_aug'):
            s, e = anno['span'][0]
        else:
            # max 32 frames / 2 fps
            s, e = parse_span(anno['span'][0], anno['duration'], 16)

            # apply temporal jittering
            offset = (e - s) / 4
            s = random.uniform(s - offset, s + offset)
            e = random.uniform(e - offset, e + offset)

            # clamp the augmented span
            s, e = parse_span([s, e], anno['duration'])

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'video_start': s,
                'video_end': e,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28,
                'max_frames': 32,
                'fps': 2.0
            }, {
                'type': 'text',
                'text': question
            }]
        }, {
            'role': 'assistant',
            'content': answer
        }]

        meta = dict(messages=messages)
        return meta
