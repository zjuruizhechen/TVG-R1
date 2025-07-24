# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy

import nncore
import torch

from videomind.constants import GROUNDER_PROMPT
from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_gnd_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.jsonl')
    else:
        pred_path = nncore.join(args.pred_path, 'output.jsonl')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    annos = DATASETS.get(args.dataset).load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = dict()

        video_path, query, duration, span = anno['video_path'], anno['query'], anno.get('duration'), anno.get('span')

        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)

        print()
        print(video_path)
        print(duration)
        print(query)

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': args.num_threads,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 150,
                'fps': 1.0
            }, {
                'type': 'text',
                'text': GROUNDER_PROMPT.format(query)
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        print(text)

        images, videos = process_vision_info(messages)

        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        output_ids = model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=256)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]

        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]

        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        print(response)

        grounder_success = len(model.reg) > 0

        if grounder_success:
            # 1. extract timestamps and confidences
            blob = model.reg[0].cpu().float()
            pred, conf = blob[:, :2] * duration, blob[:, 2:]
            print(pred[0], span, duration)

            # 2. clamp timestamps
            pred = pred.clamp(min=0, max=duration)

            # 3. round timestamps to units
            unit = getattr(DATASETS.get(args.dataset), 'UNIT', 0.001)
            pred = torch.round(pred / unit).long() * unit

            # 4. sort timestamps
            inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
            pred[inds] = pred[inds].roll(1)

            # 5. merge timestamps back with confidences
            pred = torch.cat((pred, conf), dim=1)
        else:
            print('WARNING: Failed to parse grounder response')
            pred = torch.Tensor([[0, duration, 1]])

        print(pred[0], span, duration)

        dump['vid'] = anno['vid']
        dump['qid'] = anno['qid']
        dump['pred_relevant_windows'] = pred.tolist()

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
