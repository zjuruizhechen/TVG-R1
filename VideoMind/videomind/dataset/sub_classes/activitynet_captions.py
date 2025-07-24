# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from collections import OrderedDict

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='activitynet_captions')
class ActivitynetCaptionsDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/activitynet_captions/train.json'
    ANNO_PATH_VALID = 'data/activitynet_captions/val_1.json'
    ANNO_PATH_TEST = 'data/activitynet_captions/val_2.json'

    VIDEO_ROOT = 'data/activitynet/videos_3fps_480_noaudio'
    DURATIONS = 'data/activitynet/durations.json'

    UNIT = 0.01

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID, object_pairs_hook=OrderedDict)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST, object_pairs_hook=OrderedDict)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for vid, raw_anno in raw_annos.items():
            for query, span in zip(raw_anno['sentences'], raw_anno['timestamps']):
                anno = dict(
                    source='activitynet_captions',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    duration=durations[vid],
                    query=parse_query(query),
                    span=[span])

                annos.append(anno)

        return annos


@DATASETS.register(name='activitynet_captions_bias')
class ActivitynetCaptionsBiasDataset(ActivitynetCaptionsDataset):

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID, object_pairs_hook=OrderedDict)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST, object_pairs_hook=OrderedDict)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for vid, raw_anno in raw_annos.items():
            assert len(raw_anno['sentences']) == len(raw_anno['timestamps'])

            for i in range(len(raw_anno['sentences']) - 1):
                span_a = raw_anno['timestamps'][i]
                span_b = raw_anno['timestamps'][i + 1]

                if span_b[0] - span_a[1] < 3:
                    query_a = parse_query(f"The moment before {raw_anno['sentences'][i + 1]}")
                    query_b = parse_query(f"The moment after {raw_anno['sentences'][i]}")

                    anno_a = dict(
                        source='activitynet_captions_bias',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                        duration=durations[vid],
                        query=query_a,
                        span=[span_a])

                    anno_b = dict(
                        source='activitynet_captions_bias',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                        duration=durations[vid],
                        query=query_b,
                        span=[span_b])

                    annos.append(anno_a)
                    annos.append(anno_b)

        return annos
