# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore
from torch.utils.data import Dataset

from videomind.dataset.hybrid import DATASETS
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='mlvu')
class MLVUDataset(Dataset):

    TASK_TO_DIR_MAP = {
        'plotQA': '1_plotQA',
        'findNeedle': '2_needle',
        'ego': '3_ego',
        'count': '4_count',
        'order': '5_order',
        'anomaly_reco': '6_anomaly_reco',
        'topic_reasoning': '7_topic_reasoning'
    }

    DATA_ROOT = 'data/mlvu'

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        paths = [nncore.join(self.DATA_ROOT, 'json', f'{n}.json') for n in self.TASK_TO_DIR_MAP.values()]

        raw_annos = nncore.flatten([nncore.load(p) for p in paths])

        annos = []
        for raw_anno in raw_annos:
            task = raw_anno['question_type']
            video_name = nncore.join(self.TASK_TO_DIR_MAP[task], raw_anno['video'])

            options = raw_anno['candidates']
            answer = raw_anno['answer']
            ans = chr(ord('A') + options.index(answer))

            anno = dict(
                source='mlvu',
                data_type='multimodal',
                video_path=nncore.join(self.DATA_ROOT, 'video', video_name),
                query=parse_query(raw_anno['question']),
                question=parse_question(raw_anno['question']),
                options=options,
                answer=answer,
                ans=ans,
                task=task)

            annos.append(anno)

        return annos
