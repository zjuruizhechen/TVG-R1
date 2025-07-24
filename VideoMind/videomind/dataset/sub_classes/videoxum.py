# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='videoxum')
class VideoXumDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/videoxum/train_videoxum.json'
    ANNO_PATH_VALID = 'data/videoxum/val_videoxum.json'
    ANNO_PATH_TEST = 'data/videoxum/test_videoxum.json'

    VIDEO_ROOT = 'data/activitynet/videos_3fps_480_noaudio'

    UNIT = 0.01

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_id']

            duration = raw_anno['duration']

            for query, spans in zip(raw_anno['tsum'], raw_anno['vsum']):
                assert len(spans) == 10

                # average the spans from 10 annotators
                span = [round(sum(s[0] for s in spans) / 10, 2), round(sum(s[1] for s in spans) / 10, 2)]

                anno = dict(
                    source='videoxum',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    duration=duration,
                    query=parse_query(query),
                    span=[span])

                annos.append(anno)

            annos.append(anno)

        return annos
