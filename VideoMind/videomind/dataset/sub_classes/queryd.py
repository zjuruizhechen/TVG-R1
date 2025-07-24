# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='queryd')
class QuerYDDataset(GroundingDataset):

    VID_PATH = 'data/queryd/train_list.txt'
    QUERY_PATH = 'data/queryd/raw_captions_combined_filtered-v2.pkl'
    SPAN_PATH = 'data/queryd/times_captions_combined_filtered-v2.pkl'

    VIDEO_ROOT = 'data/queryd/videos_3fps_480_noaudio'
    DURATIONS = 'data/queryd/durations.json'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        vids = nncore.load(self.VID_PATH)
        queries = nncore.load(self.QUERY_PATH)
        spans = nncore.load(self.SPAN_PATH)
        durations = nncore.load(self.DURATIONS)

        annos = []
        for vid in vids:
            for query, span in zip(queries[vid], spans[vid]):
                video_name = vid[6:]

                if video_name not in durations:
                    continue

                anno = dict(
                    source='queryd',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, video_name + '.mp4'),
                    duration=durations[video_name],
                    query=parse_query(' '.join(query)),
                    span=[span])

                annos.append(anno)

        return annos
