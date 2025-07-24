# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import AnsweringCropDataset, AnsweringDataset, GroundingDataset
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='qa_ego4d')
class QAEgo4DDataset(AnsweringDataset):

    ANNO_PATH_TRAIN = 'data/qa_ego4d/annotations.QaEgo4D_train.json'
    ANNO_PATH_VALID = 'data/qa_ego4d/annotations.QaEgo4D_val_options.json'
    ANNO_PATH_TEST = 'data/qa_ego4d/annotations.QaEgo4D_test_options.json'

    VIDEO_ROOT = 'data/ego4d/v1/videos_3fps_480_noaudio'
    DURATIONS = 'data/ego4d/v1/durations.json'

    SOURCE = 'qa_ego4d'
    DATA_TYPE = 'multimodal'

    UNIT = 0.001

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
            vid = raw_anno['video_id']

            duration = durations[vid]

            # too short or too long samples
            if split == 'train' and (duration < 10 or duration > 600):
                continue

            span = [raw_anno['moment_start_frame'] / 30, raw_anno['moment_end_frame'] / 30]
            span = [round(span[0], 3), round(span[1], 3)]

            # skip samples with too short moments
            # if split == 'train' and span[1] - span[0] < 2:
            #     continue

            answer = raw_anno['answer'].capitalize()

            if 'options' in raw_anno:
                options = [o.capitalize() for o in raw_anno['options']]
                idx = options.index(answer)
                ans = chr(ord('A') + idx)
            else:
                # NOTE: indeterministic evaluation
                assert len(raw_anno['wrong_answers']) == 3
                idx = random.randint(0, 3)
                ans = chr(ord('A') + idx)
                options = [o.capitalize() for o in raw_anno['wrong_answers']]
                options.insert(idx, answer)

            assert len(options) == 4, options

            anno = dict(
                source=self.SOURCE,
                data_type=self.DATA_TYPE,
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=duration,
                query=parse_query(raw_anno['question'].capitalize()),
                question=parse_question(raw_anno['question'].capitalize()),
                options=options,
                answer=answer,
                ans=ans,
                span=[span])

            annos.append(anno)

        return annos


@DATASETS.register(name='qa_ego4d_crop')
class QAEgo4DCropDataset(AnsweringCropDataset, QAEgo4DDataset):

    SOURCE = 'qa_ego4d_crop'


@DATASETS.register(name='qa_ego4d_grounding')
class QAEgo4DGroundingDataset(GroundingDataset, QAEgo4DDataset):

    SOURCE = 'qa_ego4d_grounding'
    DATA_TYPE = 'grounding'
