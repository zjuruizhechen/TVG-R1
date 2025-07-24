# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy

import nncore
import torch
from nncore.ops import temporal_iou
from torch.utils.data import Dataset

from videomind.constants import VERIFIER_PROMPT
from videomind.dataset.hybrid import DATASETS
from videomind.utils.parser import parse_span


class VerifyingDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args):
        super(VerifyingDataset, self).__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = len(anno['query'].split(' '))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if data_args.min_video_len >= 0 and anno.get('duration', float('inf')) < data_args.min_video_len:
                continue
            if data_args.max_video_len >= 0 and anno.get('duration', 0) > data_args.max_video_len:
                continue
            annos.append(anno)

        self.annos = annos
        self.raw_length = len(raw_annos)
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def __len__(self):
        return len(self.annos)

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        if nncore.is_dir(self.ANNO_PATH):
            raw_paths = nncore.ls(self.ANNO_PATH, ext='json', join_path=True, sort=True)
            raw_annos = nncore.flatten([nncore.load(p) for p in raw_paths])
        else:
            raw_annos = nncore.load(self.ANNO_PATH)

        annos = []
        for raw_anno in raw_annos:
            # using top-5 predictions
            for pred in raw_anno['pred'][:5]:
                iou = temporal_iou(torch.Tensor([pred]), torch.Tensor(raw_anno['span']))
                iou = torch.where(iou.isfinite(), iou, 0)
                iou = iou.max().item()

                positive = iou >= 0.5

                anno = dict(
                    source=self.SOURCE,
                    data_type='multimodal',
                    video_path=raw_anno['video_path'],
                    duration=raw_anno['duration'],
                    query=raw_anno['query'],
                    span=raw_anno['span'],
                    pred=pred,
                    positive=positive,
                    task=raw_anno.get('task', 'unknown'))

                annos.append(anno)

        pos_inds = [i for i, a in enumerate(annos) if a['positive']]
        neg_inds = [i for i, a in enumerate(annos) if not a['positive']]

        num_pos = len(pos_inds)
        num_neg = len(neg_inds)

        print(f'[{self.SOURCE}] pos: {num_pos} neg: {num_neg} n/p ratio: {num_neg / num_pos}')

        # filter negative samples
        # if num_neg > num_pos * 3:
        #     neg_inds = random.sample(neg_inds, int(num_pos * 3))

        # inds = pos_inds + neg_inds
        # random.shuffle(inds)
        # inds = comm.broadcast(inds)

        # annos = [annos[i] for i in inds]

        return annos

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])

        video_path, duration, query, positive = anno['video_path'], anno['duration'], anno['query'], anno['positive']

        s0, e0 = parse_span(anno['pred'], duration, 2)
        offset = (e0 - s0) / 2
        s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

        # percentage of s0, e0 within s1, e1
        s = (s0 - s1) / (e1 - s1)
        e = (e0 - s1) / (e1 - s1)

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'video_start': s1,
                'video_end': e1,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 64,
                'fps': 2.0
            }, {
                'type': 'text',
                'text': VERIFIER_PROMPT.format(query)
            }]
        }]

        messages = messages + [{'role': 'assistant', 'content': 'Yes.' if positive else 'No.'}]
        meta = dict(messages=messages, ss=s, se=e)

        return meta


@DATASETS.register(name='qvhighlights_verify_2b')
class QVHighlightsVerify2BDataset(VerifyingDataset):

    ANNO_PATH = 'data/verifying/verifying_qvhighlights_2b.json'

    SOURCE = 'qvhighlights_verify_2b'


@DATASETS.register(name='didemo_verify_2b')
class DiDeMoVerify2BDataset(VerifyingDataset):

    ANNO_PATH = 'data/verifying/verifying_didemo_2b.json'

    SOURCE = 'didemo_verify_2b'


@DATASETS.register(name='tacos_verify_2b')
class TACoSVerify2BDataset(VerifyingDataset):

    ANNO_PATH = 'data/verifying/verifying_tacos_2b.json'

    SOURCE = 'tacos_verify_2b'


@DATASETS.register(name='qvhighlights_verify_7b')
class QVHighlightsVerify7BDataset(VerifyingDataset):

    ANNO_PATH = 'data/verifying/verifying_qvhighlights_7b.json'

    SOURCE = 'qvhighlights_verify_7b'


@DATASETS.register(name='didemo_verify_7b')
class DiDeMoVerify7BDataset(VerifyingDataset):

    ANNO_PATH = 'data/verifying/verifying_didemo_7b.json'

    SOURCE = 'didemo_verify_7b'


@DATASETS.register(name='tacos_verify_7b')
class TACoSVerify7BDataset(VerifyingDataset):

    ANNO_PATH = 'data/verifying/verifying_tacos_7b.json'

    SOURCE = 'tacos_verify_7b'
