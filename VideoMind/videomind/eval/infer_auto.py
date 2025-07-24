# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import json
from contextlib import nullcontext

import nncore
import torch

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration, load_subtitle
from videomind.utils.parser import parse_query, parse_span


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--model_gnd_path')
    parser.add_argument('--model_ver_path')
    parser.add_argument('--model_pla_path')
    parser.add_argument('--model_ans_path')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--use_subtitle', action='store_true')
    parser.add_argument('--auto_rephrasing', action='store_true')
    parser.add_argument('--auto_planning', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    # NOTE:
    # 1. grounder is always true so no need to store
    # 2. answerer would always be used (when set to false, the base model would be used as the answerer)
    adapter_state = dict(planner=False, verifier=False, answerer=False)

    print('Initializing role *grounder*')
    model, processor = build_model(args.model_gnd_path, device=args.device)
    device = next(model.parameters()).device

    if args.model_pla_path is not None:
        adapter_path = nncore.join(args.model_pla_path, 'planner')
        if nncore.is_dir(adapter_path):
            print('Initializing role *planner*')
            model.load_adapter(adapter_path, adapter_name='planner')
            adapter_state['planner'] = True

    if args.model_ver_path is not None:
        adapter_path = nncore.join(args.model_ver_path, 'verifier')
        if nncore.is_dir(adapter_path):
            print('Initializing role *verifier*')
            model.load_adapter(adapter_path, adapter_name='verifier')
            adapter_state['verifier'] = True

    if args.model_ans_path is not None:
        adapter_path = nncore.join(args.model_ans_path, 'answerer')
        if nncore.is_dir(adapter_path):
            print('Initializing role *answerer*')
            model.load_adapter(adapter_path, adapter_name='answerer')
            adapter_state['answerer'] = True

    annos = DATASETS.get(args.dataset).load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)
            dump['duration'] = duration

        print()
        print(video_path)
        print(duration)

        # sometimes the sample is for grounding only
        do_answering = all(k in anno for k in ('question', 'options'))

        if do_answering:
            question, options, ans = anno['question'], anno['options'], anno['ans']

            if args.style in ('mcq', 'options'):
                prompt = question + '\nOptions:'
                for idx, opt in enumerate(options):
                    prompt += f"\n({chr(ord('A') + idx)}) {opt.capitalize()}"
                prompt += '\nPlease only give the best option.'
            else:
                prompt = question

            print(prompt)
            print(options)
            print(ans)
        else:
            question = anno['query']
            print(question)

        # do grounding by default
        do_grounding = True

        # initialize grounding query as question
        query = question

        # initialize agent list
        dump['agents'] = []

        if adapter_state['planner'] and (args.auto_rephrasing or args.auto_planning):
            print('=============== planner ===============')

            dump['agents'].append('planner')

            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': args.num_threads,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 100,
                    'fps': 1.0
                }, {
                    'type': 'text',
                    'text': PLANNER_PROMPT.format(question)
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            print(text)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(device)

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('planner')

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

            dump['planner_response'] = response

            try:
                parsed = json.loads(response)
                action = parsed[0] if isinstance(parsed, list) else parsed
                if args.auto_rephrasing and action['type'].lower() == 'grounder' and action['value']:
                    query = action['value']
                    dump['planner_parsed_query'] = query
                elif args.auto_planning and action['type'].lower() == 'answerer':
                    do_grounding = False
            except Exception:
                print('WARNING: Failed to parse planner response')

        if do_grounding:
            print('=============== grounder ===============')

            dump['agents'].append('grounder')

            query = parse_query(query)

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

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('grounder')

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

            dump['grounder_response'] = response
            dump['grounder_success'] = len(model.reg) > 0

            if dump['grounder_success']:
                # 1. extract timestamps and confidences
                blob = model.reg[0].cpu().float()
                pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()

                # 2. clamp timestamps
                pred = pred.clamp(min=0, max=duration)

                # 3. round timestamps to units
                unit = getattr(DATASETS.get(args.dataset), 'UNIT', 0.001)
                pred = torch.round(pred / unit).long() * unit

                # 4. sort timestamps
                inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
                pred[inds] = pred[inds].roll(1)

                # 5. convert timestamps to list
                pred = pred.tolist()
            else:
                print('WARNING: Failed to parse grounder response')

                if adapter_state['verifier']:
                    pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
                    conf = [0] * 5
                else:
                    pred = [[0, duration]]
                    conf = [0]

            print(pred[0], span, duration)
            dump['pred'] = pred
            dump['conf'] = conf

        if do_grounding and adapter_state['verifier'] and len(pred) > 1:
            print('=============== verifier ===============')

            dump['agents'].append('verifier')

            # using top-5 predictions
            probs = []
            for cand in pred[:5]:
                s0, e0 = parse_span(cand, duration, 2)
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
                        'num_threads': args.num_threads,
                        'video_start': s1,
                        'video_end': e1,
                        'min_pixels': 36 * 28 * 28,
                        'max_pixels': 64 * 28 * 28,
                        'max_frames': 64,
                        'fps': 2.0
                    }, {
                        'type': 'text',
                        'text': VERIFIER_PROMPT.format(question)
                    }]
                }]

                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                print(text)
                images, videos = process_vision_info(messages)
                data = processor(text=[text], images=images, videos=videos, return_tensors='pt')

                # ===== insert segment start/end tokens =====
                video_grid_thw = data['video_grid_thw'][0]
                num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
                assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

                pos_s, pos_e = round(s * num_frames), round(e * num_frames)
                pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
                assert pos_s <= pos_e, (num_frames, s, e)

                base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
                pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

                input_ids = data['input_ids'][0].tolist()
                input_ids.insert(pos_s, model.config.seg_s_token_id)
                input_ids.insert(pos_e, model.config.seg_e_token_id)
                data['input_ids'] = torch.LongTensor([input_ids])
                data['attention_mask'] = torch.ones_like(data['input_ids'])
                # ===========================================

                data = data.to(device)

                model.base_model.disable_adapter_layers()
                model.base_model.enable_adapter_layers()
                model.set_adapter('verifier')

                with torch.inference_mode():
                    logits = model(**data).logits[0, -1].softmax(dim=-1)

                # NOTE: magic numbers here
                # In Qwen2-VL vocab: 9454 -> Yes, 2753 -> No
                score = (logits[9454] - logits[2753]).sigmoid().item()
                probs.append(score)

            ranks = torch.Tensor(probs).argsort(descending=True).tolist()
            print(probs)
            print(ranks)

            pred = [pred[idx] for idx in ranks]
            conf = [conf[idx] for idx in ranks]
            print(pred[0], span, duration)

            dump['probs'] = probs
            dump['ranks'] = ranks
            dump['pred_ori'] = dump['pred']
            dump['conf_ori'] = dump['conf']
            dump['pred'] = pred
            dump['conf'] = conf

        if do_answering:
            print('=============== answerer ===============')

            dump['agents'].append('answerer')

            # choose the potential best moment
            selected = pred[0] if 'pred' in dump else [0, duration]

            min_len = getattr(DATASETS.get(args.dataset), 'MIN_LEN', 32)
            s, e = parse_span(selected, duration, min_len)
            print([s, e], span, duration)

            if args.use_subtitle and 'subtitle_path' in anno and nncore.is_file(anno['subtitle_path']):
                subs = load_subtitle(anno['subtitle_path'])
                subs = [f'{round(a - s, 1)}s - {round(b - s, 1)}s, {t}\n' for a, b, t in subs if a >= s and b <= e]
                # use only the first 100 subtitles to save memory
                subs = ''.join(subs[:100])
                prompt = f'You are given a video with {round(e - s, 1)} seconds long.\nSubtitles:\n{subs}' + prompt

            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': args.num_threads,
                    'video_start': s,
                    'video_end': e,
                    'min_pixels': 128 * 28 * 28,
                    'max_pixels': 256 * 28 * 28,
                    'max_frames': 32,
                    'fps': 2.0
                }, {
                    'type': 'text',
                    'text': prompt
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            text += 'Best Option: (' if args.style == 'mcq' else ''
            print(text)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(device)

            if adapter_state['answerer']:
                model.base_model.disable_adapter_layers()
                model.base_model.enable_adapter_layers()
                model.set_adapter('answerer')
                context = nullcontext
            else:
                context = model.disable_adapter

            with context():
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

            dump['answerer_response'] = response
            dump['response'] = response

        dumps.append(dump)

    nncore.dump(dumps, pred_path)