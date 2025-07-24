# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy

import nncore
from torch.utils.data import Dataset

from videomind.constants import PLANNER_PROMPT
from videomind.dataset.hybrid import DATASETS


class PlanningDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args):
        super(PlanningDataset, self).__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = len(anno.get('question', '').split(' ')) + len(anno.get('query', '').split(' '))
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

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'
        annos = nncore.load(self.ANNO_PATH)
        return annos

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])

        video_path, route, question, query = anno['video_path'], anno['route'], anno['question'], anno.get('query')

        if route == 1:
            # rephrasing + grounding + answering
            response = f'[{{"type": "grounder", "value": "{query}"}}, {{"type": "verifier"}}, {{"type": "answerer"}}]'
        elif route == 2:
            # grounding + answering
            response = f'[{{"type": "grounder", "value": "{question}"}}, {{"type": "verifier"}}, {{"type": "answerer"}}]'
        elif route == 3:
            # rephrasing + grounding
            response = f'[{{"type": "grounder", "value": "{query}"}}, {{"type": "verifier"}}]'
        elif route == 4:
            # answering
            response = '[{"type": "answerer"}]'
        else:
            raise KeyError(f'unknown route type: {route}')

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 100,
                'fps': 1.0
            }, {
                'type': 'text',
                'text': PLANNER_PROMPT.format(question)
            }]
        }, {
            'role': 'assistant',
            'content': response
        }]

        meta = dict(messages=messages)
        return meta


@DATASETS.register(name='mixed_planning')
class MixedPlanningDataset(PlanningDataset):

    ANNO_PATH = 'data/planning/planning_nextqa_qvhighlights_gpt4o_mini.jsonl'
