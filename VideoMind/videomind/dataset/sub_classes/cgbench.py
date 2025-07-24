# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore
from torch.utils.data import Dataset

from videomind.dataset.hybrid import DATASETS
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='cgbench')
class CGBenchDataset(Dataset):

    ANNO_PATH_TEST = 'data/cgbench/cgbench_mini.json'

    VIDEO_ROOT = 'data/cgbench/videos_3fps_480_noaudio'
    SUBTITLE_ROOT = 'data/cgbench/subtitles'
    DURATIONS = 'data/cgbench/durations.json'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        raw_annos = nncore.load(self.ANNO_PATH_TEST)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_uid']

            anno = dict(
                source='cgbench',
                data_type='multimodal',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                subtitle_path=nncore.join(self.SUBTITLE_ROOT, vid + '.srt'),
                duration=durations[vid],
                query=parse_query(raw_anno['question']),
                question=parse_question(raw_anno['question']),
                options=[o.capitalize() for o in raw_anno['choices']],
                answer=raw_anno['answer'].capitalize(),
                ans=raw_anno['right_answer'],
                span=raw_anno['clue_intervals'],
                task=raw_anno['sub_category'],
                domain=raw_anno['domain'])

            annos.append(anno)

        return annos
