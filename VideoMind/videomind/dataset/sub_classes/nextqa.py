# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import csv

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import AnsweringDataset
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='nextqa')
class NExTQADataset(AnsweringDataset):

    ANNO_PATH_TRAIN = 'data/nextqa/train.csv'
    ANNO_PATH_VALID = 'data/nextqa/val.csv'
    ANNO_PATH_TEST = 'data/nextqa/test.csv'

    VIDEO_ID_MAP = 'data/nextqa/map_vid_vidorID.json'
    VIDEO_ROOT = 'data/nextqa/NExTVideo'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            anno_path = self.ANNO_PATH_TRAIN
        elif split == 'valid':
            anno_path = self.ANNO_PATH_VALID
        else:
            anno_path = self.ANNO_PATH_TEST

        with open(anno_path, mode='r') as f:
            reader = csv.DictReader(f)
            raw_annos = [d for d in reader]

        video_id_map = nncore.load(self.VIDEO_ID_MAP)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video']
            qid = raw_anno['qid']

            video_id = video_id_map[vid]
            query = parse_query(raw_anno['question'].capitalize() + '?')
            question = parse_question(raw_anno['question'].capitalize() + '?')
            options = [raw_anno[k].capitalize() for k in ('a0', 'a1', 'a2', 'a3', 'a4')]
            ans = chr(ord('A') + int(raw_anno['answer']))
            answer = options[int(raw_anno['answer'])]

            anno = dict(
                source='nextqa',
                data_type='multimodal',
                uid=f'{vid}_{qid}',
                video_path=nncore.join(self.VIDEO_ROOT, video_id + '.mp4'),
                query=query,
                question=question,
                options=options,
                answer=answer,
                ans=ans,
                task=raw_anno['type'])

            annos.append(anno)

        return annos
