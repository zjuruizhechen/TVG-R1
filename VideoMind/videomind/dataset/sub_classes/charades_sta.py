# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='charades_sta')
class CharadesSTADataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/charades_sta/charades_sta_train.txt'
    ANNO_PATH_TEST = 'data/charades_sta/charades_sta_test.txt'

    VIDEO_ROOT = 'data/charades_sta/videos_3fps_480_noaudio'
    DURATIONS = 'data/charades_sta/durations.json'

    UNIT = 0.1

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            info, query = raw_anno.split('##')
            vid, s, e = info.split()

            anno = dict(
                source='charades_sta',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=durations[vid],
                query=parse_query(query),
                question=parse_query(query),
                span=[[float(s), float(e)]])

            annos.append(anno)

        return annos
