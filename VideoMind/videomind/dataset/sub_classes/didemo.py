# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random

import nncore
import numpy as np

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='didemo')
class DiDeMoDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/didemo/train_data.json'
    ANNO_PATH_VALID = 'data/didemo/val_data.json'
    ANNO_PATH_TEST = 'data/didemo/test_data.json'

    VIDEO_ROOT = 'data/didemo/videos_3fps_480_noaudio'
    DURATIONS = 'data/didemo/durations.json'

    UNIT = 1.0

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video'].split('.')[0]

            # apply mean on multiple spans
            span = np.array(raw_anno['times']).mean(axis=0).tolist()
            span = [round(span[0] * 5), round((span[1] + 1) * 5)]

            # augment spans during training
            if split == 'train':
                offset = random.randint(-2, 2)
                span = [span[0] + offset, span[1] + offset]

            anno = dict(
                source='didemo',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=durations[vid],
                query=parse_query(raw_anno['description']),
                span=[span])

            annos.append(anno)

        return annos
