# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import AnsweringCropDataset
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='star')
class STARDataset(AnsweringCropDataset):

    ANNO_PATH_TRAIN = 'data/star/STAR_train.json'
    ANNO_PATH_VALID = 'data/star/STAR_val.json'

    VIDEO_ROOT = 'data/charades_sta/videos_3fps_480_noaudio'
    DURATIONS = 'data/charades_sta/durations.json'

    UNIT = 0.1

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_VALID)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_id']

            options = [c['choice'] for c in raw_anno['choices']]
            answer = raw_anno['answer']
            ans = chr(ord('A') + options.index(answer))

            anno = dict(
                source='star',
                data_type='multimodal',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=durations[vid],
                query=parse_query(raw_anno['question']),
                question=parse_question(raw_anno['question']),
                options=options,
                answer=answer,
                ans=ans,
                span=[[raw_anno['start'], raw_anno['end']]],
                task=raw_anno['question_id'].split('_')[0],
                no_aug=True)

            annos.append(anno)

        return annos
