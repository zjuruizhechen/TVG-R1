# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from collections import OrderedDict

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='youcook2')
class YouCook2Dataset(GroundingDataset):

    ANNO_PATH = 'data/youcook2/youcookii_annotations_trainval.json'

    VIDEO_ROOT = 'data/youcook2/videos_3fps_480_noaudio'

    UNIT = 1.0

    @classmethod
    def load_annos(self, split='train'):
        subset = 'training' if split == 'train' else 'validation'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)['database']

        all_videos = nncore.ls(self.VIDEO_ROOT, ext='.mp4')
        all_videos = set(v[:11] for v in all_videos)

        annos = []
        for vid, raw_anno in raw_annos.items():
            if raw_anno['subset'] != subset:
                continue

            if vid not in all_videos:
                continue

            duration = raw_anno['duration']

            for meta in raw_anno['annotations']:
                anno = dict(
                    source='youcook2',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    duration=duration,
                    query=parse_query(meta['sentence']),
                    span=[meta['segment']])

                annos.append(anno)

            annos.append(anno)

        return annos


@DATASETS.register(name='youcook2_bias')
class YouCook2BiasDataset(YouCook2Dataset):

    @classmethod
    def load_annos(self, split='train'):
        subset = 'training' if split == 'train' else 'validation'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)['database']

        all_videos = nncore.ls(self.VIDEO_ROOT, ext='.mp4')
        all_videos = set(v[:11] for v in all_videos)

        annos = []
        for vid, raw_anno in raw_annos.items():
            if raw_anno['subset'] != subset:
                continue

            if vid not in all_videos:
                continue

            duration = raw_anno['duration']

            moments = raw_anno['annotations']

            for i in range(len(moments) - 1):
                span_a = moments[i]['segment']
                span_b = moments[i + 1]['segment']

                if span_b[0] - span_a[1] < 3:
                    query_a = parse_query(f"The moment before {moments[i + 1]['sentence']}")
                    query_b = parse_query(f"The moment after {moments[i]['sentence']}")

                    anno_a = dict(
                        source='youcook2_bias',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                        duration=duration,
                        query=parse_query(query_a),
                        span=[span_a])

                    anno_b = dict(
                        source='youcook2_bias',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                        duration=duration,
                        query=parse_query(query_b),
                        span=[span_b])

                    annos.append(anno_a)
                    annos.append(anno_b)

        return annos
