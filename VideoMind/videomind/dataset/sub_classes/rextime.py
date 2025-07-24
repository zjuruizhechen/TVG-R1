# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import AnsweringCropDataset, AnsweringDataset, GroundingDataset
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='rextime')
class ReXTimeDataset(AnsweringDataset):

    ANNO_PATH_TRAIN = 'rextime/rextime_train.json'
    ANNO_PATH_VALID = '/opt/tiger/video-r1/VideoMind/rextime/rextime_val.json'
    ANNO_PATH_TEST = '/opt/tiger/video-r1/VideoMind/rextime/rextime_test_release.json'

    VIDEO_ROOT_ANET = 'data/activitynet/videos_3fps_480_noaudio'
    VIDEO_ROOT_QVHL = 'data/qvhighlights/videos_3fps_480_noaudio'

    DURATIONS_ANET = 'data/activitynet/durations.json'
    DURATIONS_QVHL = 'data/qvhighlights/durations.json'

    SOURCE = 'rextime'
    DATA_TYPE = 'multimodal'

    UNIT = 1.0
    MIN_LEN = 64

    @classmethod
    def load_annos(self, split='valid'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            print('WARNING: Test split does not have ground truth annotations')
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        durations_anet = nncore.load(self.DURATIONS_ANET)
        durations_qvhl = nncore.load(self.DURATIONS_QVHL)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['vid']

            if len(vid) == 13:
                video_path = nncore.join(self.VIDEO_ROOT_ANET, vid + '.mp4')
                duration = durations_anet[vid]
            else:
                video_path = nncore.join(self.VIDEO_ROOT_QVHL, vid + '.mp4')
                duration = durations_qvhl[vid]
            anno = dict(
                source=self.SOURCE,
                data_type=self.DATA_TYPE,
                video_path=video_path,
                duration=duration,
                query=parse_query(raw_anno['question']),
                question=parse_question(raw_anno['question']),
                options=[o.capitalize() for o in raw_anno['options']],
                answer=raw_anno['answer'].replace('From <s0> to <e0>, ', '').capitalize(),
                ans=raw_anno['ans'],
                span=[raw_anno['span']],
                task=raw_anno['category'])

            annos.append(anno)

        return annos


@DATASETS.register(name='rextime_crop')
class ReXTimeCropDataset(AnsweringCropDataset, ReXTimeDataset):

    SOURCE = 'rextime_crop'


@DATASETS.register(name='rextime_grounding')
class ReXTimeGroundingDataset(GroundingDataset, ReXTimeDataset):

    SOURCE = 'rextime_grounding'
    DATA_TYPE = 'grounding'
