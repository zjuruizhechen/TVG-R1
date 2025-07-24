# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from collections import OrderedDict

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='hirest_grounding')
class HiRESTGroundingDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/hirest/all_data_train.json'
    ANNO_PATH_VALID = 'data/hirest/all_data_val.json'

    VIDEO_ROOT = 'data/hirest/videos_3fps_480_noaudio'

    UNIT = 1.0

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_VALID, object_pairs_hook=OrderedDict)

        all_videos = nncore.ls(self.VIDEO_ROOT, ext='.mp4')
        all_videos = set(v[:11] for v in all_videos)

        annos = []
        for query, videos in raw_annos.items():
            for video_name, raw_anno in videos.items():
                if not raw_anno['relevant'] or not raw_anno['clip']:
                    continue

                assert len(raw_anno['bounds']) == 2

                vid = video_name.split('.')[0]

                if vid not in all_videos:
                    continue

                anno = dict(
                    source='hirest_grounding',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, video_name),
                    duration=raw_anno['v_duration'],
                    query=parse_query(query),
                    span=[raw_anno['bounds']])

                annos.append(anno)

        return annos


@DATASETS.register(name='hirest_step')
class HiRESTStepDataset(HiRESTGroundingDataset):

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_VALID, object_pairs_hook=OrderedDict)

        all_videos = nncore.ls(self.VIDEO_ROOT, ext='.mp4')
        all_videos = set(v[:11] for v in all_videos)

        annos = []
        for query, videos in raw_annos.items():
            for video_name, raw_anno in videos.items():
                if not raw_anno['relevant'] or not raw_anno['clip'] or len(raw_anno['steps']) == 0:
                    continue

                vid = video_name.split('.')[0]

                if vid not in all_videos:
                    continue

                for step in raw_anno['steps']:
                    assert len(step['absolute_bounds']) == 2

                    anno = dict(
                        source='hirest_step',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, video_name),
                        duration=raw_anno['v_duration'],
                        query=parse_query(step['heading']),
                        span=[step['absolute_bounds']])

                    annos.append(anno)

        return annos


@DATASETS.register(name='hirest_step_bias')
class HiRESTStepBiasDataset(HiRESTStepDataset):

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_VALID, object_pairs_hook=OrderedDict)

        all_videos = nncore.ls(self.VIDEO_ROOT, ext='.mp4')
        all_videos = set(v[:11] for v in all_videos)

        annos = []
        for query, videos in raw_annos.items():
            for video_name, raw_anno in videos.items():
                if not raw_anno['relevant'] or not raw_anno['clip'] or len(raw_anno['steps']) == 0:
                    continue

                vid = video_name.split('.')[0]

                if vid not in all_videos:
                    continue

                for i in range(len(raw_anno['steps']) - 1):
                    span_a = raw_anno['steps'][i]['absolute_bounds']
                    span_b = raw_anno['steps'][i + 1]['absolute_bounds']

                    assert len(span_a) == 2 and len(span_b) == 2 and span_a[1] == span_b[0]

                    query_a = parse_query(f"The moment before {raw_anno['steps'][i + 1]['heading']}")
                    query_b = parse_query(f"The moment after {raw_anno['steps'][i]['heading']}")

                    anno_a = dict(
                        source='hirest_step_bias',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, video_name),
                        duration=raw_anno['v_duration'],
                        query=query_a,
                        span=[span_a])

                    anno_b = dict(
                        source='hirest_step_bias',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, video_name),
                        duration=raw_anno['v_duration'],
                        query=query_b,
                        span=[span_b])

                    annos.append(anno_a)
                    annos.append(anno_b)

        return annos
