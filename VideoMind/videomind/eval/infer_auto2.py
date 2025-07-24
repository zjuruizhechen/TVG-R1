# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import json
from contextlib import nullcontext

import nncore
import torch

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT, GROUNDEDANSWER_PROMPT, ANSWER_PROMPT, GROUNDING_PROMPT, TIMER1_TEMPLATE, VideoR1_QUESTION_TEMPLATE
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
    parser.add_argument('--sys_prompt', default='answer')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--use_subtitle', action='store_true')
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

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_gnd_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    if "sft" in args.model_gnd_path or "Video-R1" in args.model_gnd_path:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    else:
        processor = AutoProcessor.from_pretrained(args.model_gnd_path)

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

        do_grounding = False
        do_grounding_no_think = False
        do_groundedanswering = False
        do_answering = False
        sys_prompt = None

        if args.sys_prompt == "grounding":
            do_grounding = True
        elif args.sys_prompt == "grounding_no_think":
            do_grounding_no_think = True
        elif args.sys_prompt == "grounded_answer":
            do_groundedanswering = True
        elif args.sys_prompt == "answer":
            do_answering = True

        # do_answering = all(k in anno for k in ('question', 'options')) and 'span' not in anno
        # do_groundedanswering = all(k in anno for k in ('question', 'options', 'span'))
        # do_grounding = all(k in anno for k in ('question', 'span')) and 'options' not in anno

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
            try:
                question = anno['query']
            except:
                question = anno['question']
            
            if "Time-R1" in args.model_gnd_path:
                prompt = TIMER1_TEMPLATE.replace('[EVENT]', question)
            elif "Video-R1" in args.model_gnd_path:
                prompt = VideoR1_QUESTION_TEMPLATE.format(Question=question)
            else:
                prompt = GROUNDER_PROMPT.format(question)
            sys_prompt = None

        elif do_grounding_no_think:
            try:
                question = anno['query']
            except:
                question = anno['question']
            prompt = "query: " + question
            sys_prompt = "You are a helpful assistant. Determine the precise time period related to the query. The specific time period MUST BE in the format [start time, end time] in seconds enclosed within <time> </time> tags."

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
                'video_start': s,
                'video_end': e,
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to("cuda")


        output_ids = model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=2048)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

        dump['answerer_response'] = response
        dump['response'] = response

        dumps.append(dump)

    nncore.dump(dumps, pred_path)

