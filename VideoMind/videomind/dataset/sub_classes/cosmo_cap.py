# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='cosmo_cap')
class CosMoCapDataset(GroundingDataset):

    ANNO_PATH = 'data/cosmo_cap/anno_cosmo_cap.jsonl'

    VIDEO_ROOT = 'data/cosmo_cap/videos_3fps_480_noaudio'

    UNIT = 1.0

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH)

        annos = []
        for raw_anno in raw_annos:
            anno = dict(
                source='cosmo_cap',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, raw_anno['vid'] + '.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=[raw_anno['span']])

            annos.append(anno)

        return annos
