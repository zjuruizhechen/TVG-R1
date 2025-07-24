# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore
from torch.utils.data import Dataset

from videomind.dataset.hybrid import DATASETS
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='lvbench')
class LVBenchDataset(Dataset):

    ANNO_PATH = 'data/lvbench/video_info.meta.jsonl'

    VIDEO_ROOT = 'data/lvbench/videos_3fps_480_noaudio'

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        raw_annos = nncore.load(self.ANNO_PATH)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['key']

            for meta in raw_anno['qa']:
                tok = meta['question'].split('\n')

                assert len(tok) == 5
                assert all(any(o.startswith(k) for k in ('(A) ', '(B) ', '(C) ', '(D) ')) for o in tok[1:])

                options = [o[4:] for o in tok[1:]]
                ans = meta['answer']
                answer = options[ord(ans) - ord('A')]
                assert ans in 'ABCD'

                anno = dict(
                    source='lvbench',
                    data_type='multimodal',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    query=parse_query(tok[0]),
                    question=parse_question(tok[0]),
                    options=options,
                    answer=answer,
                    ans=ans,
                    task=meta['question_type'],
                    time_reference=meta['time_reference'])

                annos.append(anno)

        return annos
