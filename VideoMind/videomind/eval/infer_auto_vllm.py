# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import json
from contextlib import nullcontext

import nncore
import torch
from tqdm import tqdm

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT, GROUNDEDANSWER_PROMPT, ANSWER_PROMPT, GROUNDING_PROMPT
from videomind.dataset.hybrid import DATASETS
from qwen_vl_utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration, load_subtitle
from videomind.utils.parser import parse_query, parse_span

from vllm import LLM, SamplingParams
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

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

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

    BSZ = 32

    llm = LLM(
            model=args.model_gnd_path,
            tensor_parallel_size=1,
            max_model_len = 8192,
            gpu_memory_utilization=0.8,
            limit_mm_per_prompt={"image": 0, "video": 1},
        )

    sampling_params = SamplingParams(
            max_tokens=512,
            stop_token_ids=[],
        )

    processor = AutoProcessor.from_pretrained(args.model_gnd_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_gnd_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

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

        # sometimes the sample is for grounding only
        do_answering = all(k in anno for k in ('question', 'options')) and 'span' not in anno
        do_groundedanswering = all(k in anno for k in ('question', 'options', 'span'))
        do_grounding = all(k in anno for k in ('question', 'span')) and 'options' not in anno

        if do_answering:
            question, options, ans = anno['question'], anno['options'], anno['ans']

            if args.style in ('mcq', 'options'):
                prompt = question + '\nOptions:'
                for idx, opt in enumerate(options):
                    prompt += f"\n({chr(ord('A') + idx)}) {opt.capitalize()}"
                prompt += '\nPlease only give the best option.'
            else:
                prompt = question
            sys_prompt = None

        elif do_grounding:
            question = anno['query']
            prompt = GROUNDER_PROMPT.format(question)
            sys_prompt = None

        elif do_groundedanswering:
            question, options, span, ans = anno['question'], anno['options'], anno['span'], anno['ans']
            prompt = question + '\nOptions:'
            for idx, opt in enumerate(options):
                    prompt += f"\n({chr(ord('A') + idx)}) {opt.capitalize()}"
            if "step" in args.model_gnd_path:
                if "GVQA" in args.model_gnd_path:
                    sys_prompt = GROUNDEDANSWER_PROMPT
                elif "VQA" in args.model_gnd_path:
                    sys_prompt = ANSWER_PROMPT
                else:
                    sys_prompt = GROUNDING_PROMPT
            else:
                sys_prompt = GROUNDEDANSWER_PROMPT


        # initialize grounding query as question
        query = question

        print('=============== answerer ===============')

        # choose the potential best moment
        selected = pred[0] if 'pred' in dump else [0, duration]

        min_len = getattr(DATASETS.get(args.dataset), 'MIN_LEN', 32)
        s, e = parse_span(selected, duration, min_len)
        # print([s, e], span, duration)

        if args.use_subtitle and 'subtitle_path' in anno and nncore.is_file(anno['subtitle_path']):
            # use only the first 100 subtitles to save memory
            subs = load_subtitle(anno['subtitle_path'])[:100]
            subs = [f'{round(a - s, 1)}s - {round(b - s, 1)}s, {t}\n' for a, b, t in subs if a >= s and b <= e]
            subs = ''.join(subs)
            prompt = f'You are given a video with {round(e - s, 1)} seconds long.\nSubtitles:\n{subs}' + prompt

        print(prompt)
        if sys_prompt is None:
            sys_prompt = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": [
                {'type': 'text',
                'text': sys_prompt}]}, 
            {'role': 'user',
            'content': [
                {
                'type': 'video',
                'video': video_path,
                'num_threads': args.num_threads,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 512 * 28 * 28,
                'nframes': 32,
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

        dump['messages'] = messages
        dumps.append(dump)

    start_idx = 0
    for i in tqdm(range(start_idx, len(dumps), BSZ), desc="Processing batches"):
        batch_messages = [dump['messages'] for dump in dumps[i:i + BSZ]]

        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        

        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
            
        image_idx = 0
        video_idx = 0

        llm_inputs = []
        print(prompts[0])
        
        for idx, prompt in enumerate(prompts):
            mm_type = batch_messages[idx][0]['content'][0]['type']
            sample_mm_data = {}
            sample_video_kw = {}
            if mm_type == 'image':
                sample_mm_data["image"] = image_inputs[image_idx]
                image_idx += 1
            elif mm_type == 'video':
                sample_mm_data["video"] = video_inputs[video_idx]
                for key, value in video_kwargs.items():
                    sample_video_kw[key] = value[video_idx]
                video_idx += 1
                    
            
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": sample_mm_data,
                "mm_processor_kwargs": sample_video_kw,
            })
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        batch_output_text = [out.outputs[0].text for out in outputs]


        for j, (sample, model_output) in enumerate(zip(dumps[i:i+BSZ], batch_output_text), start=i):
            sample['response'] = model_output

    nncore.dump(dumps, pred_path)



