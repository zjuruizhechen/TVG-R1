# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import AnsweringCropDataset, AnsweringDataset, GroundingDataset
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='ego_timeqa')
class EgoTimeQADataset(AnsweringDataset):

    ANNO_PATH_TRAIN = 'data/ego_timeqa/annotations.EgoTimeQA.json'

    VIDEO_ROOT = 'data/ego4d/v2/videos_3fps_480_noaudio'
    DURATIONS = 'data/ego4d/v2/durations.json'

    SOURCE = 'ego_timeqa'
    DATA_TYPE = 'multimodal'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_id']

            duration = durations[vid]

            # 303k -> 284k (to be verified)
            if duration < 10 or duration > 600:
                continue

            span = [raw_anno['moment_start_frame'] / 30, raw_anno['moment_end_frame'] / 30]
            span = [round(span[0], 3), round(span[1], 3)]

            # this would remove many samples (284k -> 37k)
            # if span[1] - span[0] < 2:
            #     continue

            question = raw_anno['question'].replace(' l ', ' I ').capitalize()
            question = parse_question(question)
            query = parse_query(question)

            # too short or too long samples
            num_words = len(query.split(' '))
            if split == 'train' and (num_words < 3 or num_words > 30):
                continue

            answer = raw_anno['answer'].capitalize()

            assert len(raw_anno['wrong_answers']) == 3
            idx = random.randint(0, 3)
            ans = chr(ord('A') + idx)
            options = [o.capitalize() for o in raw_anno['wrong_answers']]
            options.insert(idx, answer)

            anno = dict(
                source=self.SOURCE,
                data_type=self.DATA_TYPE,
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=duration,
                query=query,
                question=question,
                options=options,
                answer=answer,
                ans=ans,
                span=[span])

            annos.append(anno)

        return annos


@DATASETS.register(name='ego_timeqa_crop')
class EgoTimeQACropDataset(AnsweringCropDataset, EgoTimeQADataset):

    SOURCE = 'ego_timeqa_crop'


@DATASETS.register(name='ego_timeqa_grounding')
class EgoTimeQAGroundingDataset(GroundingDataset, EgoTimeQADataset):

    SOURCE = 'ego_timeqa_grounding'
    DATA_TYPE = 'grounding'
