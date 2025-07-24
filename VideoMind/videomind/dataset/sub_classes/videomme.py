# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore
from torch.utils.data import Dataset

import pandas as pd
from videomind.dataset.hybrid import DATASETS
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='videomme')
class VideoMMEDataset(Dataset):

    ANNO_PATH = 'data/videomme/test-00000-of-00001.parquet'

    VIDEO_ROOT = 'data/videomme/videos'
    SUBTITLE_ROOT = 'data/videomme/subtitles'

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        raw_annos = pd.read_parquet(self.ANNO_PATH).to_dict(orient='records')

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['videoID']

            options = raw_anno['options'].tolist()

            assert len(options) == 4
            assert all(any(o.startswith(k) for k in ('A. ', 'B. ', 'C. ', 'D. ')) for o in options)

            options = [o[3:] for o in options]
            ans = raw_anno['answer']
            answer = options[ord(ans) - ord('A')]
            assert ans in 'ABCD'

            anno = dict(
                source='videomme',
                data_type='multimodal',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                query=parse_query(raw_anno['question']),
                question=parse_question(raw_anno['question']),
                options=options,
                answer=answer,
                ans=ans,
                task=raw_anno['duration'])

            annos.append(anno)

        return annos
