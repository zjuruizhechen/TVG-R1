# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from collections import OrderedDict

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='ego4d_naq')
class Ego4DNaQDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/ego4d_naq/train.json'
    ANNO_PATH_VALID = 'data/ego4d_naq/val.json'
    ANNO_PATH_TEST = 'data/ego4d_naq/test.json'

    VIDEO_ROOT = 'data/ego4d/v2/videos_3fps_480_noaudio'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID, object_pairs_hook=OrderedDict)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST, object_pairs_hook=OrderedDict)

        annos = []
        for vid, raw_anno in raw_annos.items():
            duration = raw_anno['num_frames'] / raw_anno['fps']

            # 300s: 254k samples (dropped 121k samples merged 156k samples)
            # 480s: 567k samples (dropped 249k samples merged 328k samples)
            if split == 'train' and (duration < 10 or duration > 600):
                continue

            meta = dict()
            for span, query in zip(raw_anno['exact_times'], raw_anno['sentences']):
                span = [round(span[0], 3), round(span[1], 3)]

                query = parse_query(query)

                # these annotations might be from nlq
                nlq_keys = ('who', 'what', 'when', 'in what', 'did', 'where', 'how', 'i what')
                if split == 'train' and any(query.startswith(k) for k in nlq_keys):
                    continue

                # bad samples
                if split == 'train' and '#unsure' in query:
                    continue

                # too short or too long samples
                num_words = len(query.split(' '))
                if split == 'train' and (num_words < 3 or num_words > 30):
                    continue

                if query not in meta:
                    meta[query] = []

                meta[query].append(span)

            for query, span in meta.items():
                # skip samples with multiple moments
                if len(span) > 1:
                    continue

                anno = dict(
                    source='ego4d_naq',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    duration=duration,
                    query=query,
                    span=span)

                annos.append(anno)

        return annos
