# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import re
from collections import OrderedDict

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='activitynet_rtl')
class ActivitynetRTLDataset(GroundingDataset):

    ANNO_PATH_TRAIN = 'data/activitynet_rtl/activitynet_train_gpt-4-0613_temp_6_f10009.json'
    ANNO_PATH_TEST = 'data/activitynet_rtl/annot_val_1_q229.json'

    VIDEO_ROOT = 'data/activitynet/videos_3fps_480_noaudio'

    UNIT = 0.01

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)

            annos = []
            for vid, raw_anno in raw_annos.items():
                for meta in raw_anno['QA']:
                    match = re.findall(r'<(\d+(\.\d+)?)>', meta['a'])
                    span = [float(m[0]) for m in match[:2]]

                    # some samples do not have timestamps
                    if len(span) != 2:
                        continue

                    anno = dict(
                        source='activitynet_rtl',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                        duration=raw_anno['duration'],
                        query=parse_query(meta['q']),
                        span=[span])

                    annos.append(anno)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST, object_pairs_hook=OrderedDict)

            annos = []
            for raw_anno in raw_annos:
                vid = f"v_{raw_anno['vid']}"

                match = re.findall(r'<(\d+(\.\d+)?)>', raw_anno['answer'])
                span = [float(m[0]) for m in match[:2]]
                assert len(span) == 2

                anno = dict(
                    source='activitynet_rtl',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    duration=raw_anno['duration'],
                    query=parse_query(raw_anno['question']),
                    span=[span])

                annos.append(anno)

        return annos
