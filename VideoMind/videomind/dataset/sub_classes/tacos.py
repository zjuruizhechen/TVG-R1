# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='tacos')
class TACoSDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/tacos/train.jsonl'
    ANNO_PATH_VALID = 'data/tacos/val.jsonl'
    ANNO_PATH_TEST = 'data/tacos/test.jsonl'

    VIDEO_ROOT = 'data/tacos/videos_3fps_480_noaudio'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'val':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        annos = []
        for raw_anno in raw_annos:
            assert len(raw_anno['relevant_windows']) == 1

            vid = raw_anno['vid']

            anno = dict(
                source='tacos',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '-cam-002.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=raw_anno['relevant_windows'])

            annos.append(anno)

        return annos
