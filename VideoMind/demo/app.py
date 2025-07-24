# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause license.

import html
import json
import os
import random
import time

import gradio as gr
import nncore
import spaces
import torch
from huggingface_hub import snapshot_download

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration
from videomind.utils.parser import parse_query, parse_span

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

BASE_MODEL = 'model_zoo/Qwen2-VL-2B-Instruct'
BASE_MODEL_REPO = 'Qwen/Qwen2-VL-2B-Instruct'

MODEL = 'model_zoo/VideoMind-2B'
MODEL_REPO = 'yeliudev/VideoMind-2B'

TITLE = 'VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning'

BADGE = """
<h3 align="center" style="margin-top: -0.5em;">A Chain-of-LoRA Agent for Long Video Reasoning</h3>
<div style="display: flex; justify-content: center; gap: 5px; margin-bottom: -0.7em !important;">
    <a href="https://arxiv.org/abs/2503.13444" target="_blank"><img src="https://img.shields.io/badge/arXiv-2503.13444-red"></a>
    <a href="https://videomind.github.io/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
    <a href="https://huggingface.co/collections/yeliudev/videomind-67dd41f42c57f0e7433afb36" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
    <a href="https://huggingface.co/datasets/yeliudev/VideoMind-Dataset" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
    <a href="https://github.com/yeliudev/VideoMind/blob/main/README.md" target="_blank"><img src="https://img.shields.io/badge/License-BSD--3--Clause-purple"></a>
    <a href="https://github.com/yeliudev/VideoMind" target="_blank"><img src="https://img.shields.io/github/stars/yeliudev/VideoMind"></a>
</div>
"""

LOGO = '<p align="center"><img width="350" src="https://raw.githubusercontent.com/yeliudev/VideoMind/refs/heads/main/.github/logo.png"></p>'
DISC = 'VideoMind is a multi-modal agent framework that enhances video reasoning by emulating *human-like* processes, such as *breaking down tasks*, *localizing and verifying moments*, and *synthesizing answers*. This demo showcases how VideoMind-2B handles video-language tasks. Please open an <a href="https://github.com/yeliudev/VideoMind/issues/new" target="_blank">issue</a> if you meet any problems.'  # noqa

# yapf:disable
EXAMPLES = [
    [f'{PATH}/examples/4167294363.mp4', 'Why did the old man stand up?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/5012237466.mp4', 'How does the child in stripes react about the fountain?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/13887487955.mp4', 'What did the excavator do after it pushed the cement forward?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/5188348585.mp4', 'What did the person do before pouring the liquor?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/4766274786.mp4', 'What did the girl do after the baby lost the balloon?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/4742652230.mp4', 'Why is the girl pushing the boy only around the toy but not to other places?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/9383140374.mp4', 'How does the girl in pink control the movement of the claw?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/10309844035.mp4', 'Why are they holding up the phones?', ['pla', 'gnd', 'ver', 'ans']],
    [f'{PATH}/examples/pA6Z-qYhSNg_60.0_210.0.mp4', 'Different types of meat products are being cut, shaped and prepared', ['gnd', 'ver']],
    [f'{PATH}/examples/UFWQKrcbhjI_360.0_510.0.mp4', 'A man talks to the camera whilst walking along a roadside in a rural area', ['gnd', 'ver']],
    [f'{PATH}/examples/RoripwjYFp8_210.0_360.0.mp4', 'A woman wearing glasses eating something at a street market', ['gnd', 'ver']],
    [f'{PATH}/examples/h6QKDqomIPk_210.0_360.0.mp4', 'A toddler sits in his car seat, holding his yellow tablet', ['gnd', 'ver']],
    [f'{PATH}/examples/Z3-IZ3HAmIA_60.0_210.0.mp4', 'A view from the window as the plane accelerates and takes off from the runway', ['gnd', 'ver']],
    [f'{PATH}/examples/yId2wIocTys_210.0_360.0.mp4', "Temporally locate the visual content mentioned in the text query 'kids exercise in front of parked cars' within the video.", ['pla', 'gnd', 'ver']],
    [f'{PATH}/examples/rrTIeJRVGjg_60.0_210.0.mp4', "Localize the moment that provides relevant context about 'man stands in front of a white building monologuing'.", ['pla', 'gnd', 'ver']],
    [f'{PATH}/examples/DTInxNfWXVc_210.0_360.0.mp4', "Find the video segment that corresponds to the given textual query 'man with headphones talking'.", ['pla', 'gnd', 'ver']],
]
# yapf:enable

# https://github.com/gradio-app/gradio/pull/10552
JS = """
function init() {
    if (window.innerWidth >= 1536) {
        document.querySelector('main').style.maxWidth = '1536px'
    }
}
"""

if not nncore.is_dir(BASE_MODEL):
    snapshot_download(BASE_MODEL_REPO, local_dir=BASE_MODEL)

if not nncore.is_dir(MODEL):
    snapshot_download(MODEL_REPO, local_dir=MODEL)

print('Initializing role *grounder*')
model, processor = build_model(MODEL)

print('Initializing role *planner*')
model.load_adapter(nncore.join(MODEL, 'planner'), adapter_name='planner')

print('Initializing role *verifier*')
model.load_adapter(nncore.join(MODEL, 'verifier'), adapter_name='verifier')

device = torch.device('cuda')


def seconds_to_hms(seconds):
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def random_sample():
    return random.choice(EXAMPLES)


def enable_btns():
    return (gr.Button(interactive=True), ) * 3


def disable_btns():
    return (gr.Button(interactive=False), ) * 3


def update_placeholder(role):
    placeholder = 'Ask a question about the video...' if 'ans' in role else 'Write a query to search for a moment...'
    return gr.Textbox(placeholder=placeholder)


def reset_components():
    return ['pla', 'gnd', 'ver', 'ans'], 5, 0, 256


@spaces.GPU
def main(video, prompt, role, max_candidates, temperature, max_new_tokens):
    global model, processor, device

    history = []

    if not video:
        gr.Warning('Please upload a video or click [Random] to sample one.')
        return history

    if not prompt:
        gr.Warning('Please provide a prompt or click [Random] to sample one.')
        return history

    if 'gnd' not in role and 'ans' not in role:
        gr.Warning('Please at least select Grounder or Answerer.')
        return history

    if 'ver' in role and 'gnd' not in role:
        gr.Warning('Verifier cannot be used without Grounder.')
        return history

    if 'pla' in role and 'gnd' not in role and 'ver' not in role:
        gr.Warning('Planner can only be used with Grounder and Verifier.')
        return history

    history.append({'role': 'user', 'content': prompt})
    yield history

    model = model.to(device)

    duration = get_duration(video)

    # do grounding and answering by default
    do_grounding = True
    do_answering = True

    # initialize grounding query as prompt
    query = prompt

    if 'pla' in role:
        text = PLANNER_PROMPT.format(prompt)

        history.append({
            'metadata': {
                'title': '🗺️ Working as Planner...'
            },
            'role': 'assistant',
            'content': f'##### Planner Prompt:\n\n{html.escape(text)}\n\n##### Planner Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video,
                'num_threads': 1,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 100,
                'fps': 1.0
            }, {
                'type': 'text',
                'text': text
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        model.base_model.disable_adapter_layers()
        model.base_model.enable_adapter_layers()
        model.set_adapter('planner')

        output_ids = model.generate(
            **data,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

        for i, text in enumerate(response.split(' ')):
            if i == 0:
                history[-1]['content'] = history[-1]['content'].rstrip('.')
                history[-1]['content'] += text
            else:
                history[-1]['content'] += ' ' + text
            yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        try:
            parsed = json.loads(response)
            action = parsed[0] if isinstance(parsed, list) else parsed
            if action['type'].lower() == 'grounder' and action['value']:
                query = action['value']
            elif action['type'].lower() == 'answerer':
                do_grounding = False
                do_answering = True
        except Exception:
            pass

        response = 'After browsing the video and the question. My plan to figure out the answer is as follows:\n'
        step_idx = 1
        if 'gnd' in role and do_grounding:
            response += f'\n{step_idx}. Localize the relevant moment in this video using the query "<span style="color:red">{query}</span>".'
            step_idx += 1
        if 'ver' in role and do_grounding:
            response += f'\n{step_idx}. Verify the grounded moments one-by-one and select the best cancdidate.'
            step_idx += 1
        if 'ans' in role and do_answering:
            if step_idx > 1:
                response += f'\n{step_idx}. Crop the video segment and zoom-in to higher resolution.'
            else:
                response += f'\n{step_idx}. Analyze the whole video directly without cropping.'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history

    if 'gnd' in role and do_grounding:
        query = parse_query(query)

        text = GROUNDER_PROMPT.format(query)

        history.append({
            'metadata': {
                'title': '🔍 Working as Grounder...'
            },
            'role': 'assistant',
            'content': f'##### Grounder Prompt:\n\n{html.escape(text)}\n\n##### Grounder Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video,
                'num_threads': 1,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 150,
                'fps': 1.0
            }, {
                'type': 'text',
                'text': text
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        model.base_model.disable_adapter_layers()
        model.base_model.enable_adapter_layers()
        model.set_adapter('grounder')

        output_ids = model.generate(
            **data,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

        for i, text in enumerate(response.split(' ')):
            if i == 0:
                history[-1]['content'] = history[-1]['content'].rstrip('.')
                history[-1]['content'] += text
            else:
                history[-1]['content'] += ' ' + text
            yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        if len(model.reg) > 0:
            # 1. extract timestamps and confidences
            blob = model.reg[0].cpu().float()
            pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()

            # 2. clamp timestamps
            pred = pred.clamp(min=0, max=duration)

            # 3. sort timestamps
            inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
            pred[inds] = pred[inds].roll(1)

            # 4. convert timestamps to list
            pred = pred.tolist()
        else:
            if 'ver' in role:
                pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
                conf = [0] * 5
            else:
                pred = [[0, duration]]
                conf = [0]

        response = 'The candidate moments and confidence scores are as follows:\n'
        response += '\n| ID | Start Time | End Time | Confidence |'
        response += '\n| :-: | :-: | :-: | :-: |'

        for i, (p, c) in enumerate(zip(pred[:max_candidates], conf[:max_candidates])):
            response += f'\n| {i} | {seconds_to_hms(p[0])} | {seconds_to_hms(p[1])} | {c:.2f} |'

        response += f'\n\nTherefore, the target moment might happens from <span style="color:red">{seconds_to_hms(pred[0][0])}</span> to <span style="color:red">{seconds_to_hms(pred[0][1])}</span>.'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history

    if 'ver' in role and do_grounding:
        text = VERIFIER_PROMPT.format(query)

        history.append({
            'metadata': {
                'title': '📊 Working as Verifier...'
            },
            'role': 'assistant',
            'content': f'##### Verifier Prompt:\n\n{html.escape(text)}\n\n##### Verifier Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        prob = []
        for i, cand in enumerate(pred[:max_candidates]):
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
                    'video': video,
                    'num_threads': 1,
                    'video_start': s1,
                    'video_end': e1,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 64,
                    'fps': 2.0
                }, {
                    'type': 'text',
                    'text': text
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
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
            prob.append(score)

            if i == 0:
                history[-1]['content'] = history[-1]['content'].rstrip('.')[:-1]

            response = f'\nCandidate ID {i}: P(Yes) = {score:.2f}'
            for j, text in enumerate(response.split(' ')):
                history[-1]['content'] += ' ' + text if j > 0 else text
                yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        ranks = torch.Tensor(prob).argsort(descending=True).tolist()

        prob = [prob[idx] for idx in ranks]
        pred = [pred[idx] for idx in ranks]
        conf = [conf[idx] for idx in ranks]

        response = 'After verification, the candidate moments are re-ranked as follows:\n'
        response += '\n| ID | Start Time | End Time | Score |'
        response += '\n| :-: | :-: | :-: | :-: |'

        ids = list(range(len(ranks)))
        for r, p, c in zip(ranks, pred, prob):
            response += f'\n| {ids[r]} | {seconds_to_hms(p[0])} | {seconds_to_hms(p[1])} | {c:.2f} |'

        response += f'\n\nTherefore, the target moment should be from <span style="color:red">{seconds_to_hms(pred[0][0])}</span> to <span style="color:red">{seconds_to_hms(pred[0][1])}</span>.'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history

    if 'ans' in role and do_answering:
        text = f'{prompt} Please think step by step and provide your response.'

        history.append({
            'metadata': {
                'title': '📝 Working as Answerer...'
            },
            'role': 'assistant',
            'content': f'##### Answerer Prompt:\n\n{html.escape(text)}\n\n##### Answerer Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        # choose the potential best moment
        selected = pred[0] if 'gnd' in role and do_grounding else [0, duration]
        s, e = parse_span(selected, duration, 32)

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video,
                'num_threads': 1,
                'video_start': s,
                'video_end': e,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28,
                'max_frames': 32,
                'fps': 2.0
            }, {
                'type': 'text',
                'text': text
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        with model.disable_adapter():
            output_ids = model.generate(
                **data,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=max_new_tokens)

        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

        for i, text in enumerate(response.split(' ')):
            if i == 0:
                history[-1]['content'] = history[-1]['content'].rstrip('.')
                history[-1]['content'] += text
            else:
                history[-1]['content'] += ' ' + text
            yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        if 'gnd' in role and do_grounding:
            response = f'After zooming in and analyzing the target moment, I finalize my answer: <span style="color:green">{response}</span>'
        else:
            response = f'After watching the whole video, my answer is: <span style="color:green">{response}</span>'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history


def build_demo():
    chat = gr.Chatbot(
        type='messages',
        height='70em',
        resizable=True,
        avatar_images=[f'{PATH}/assets/user.png', f'{PATH}/assets/bot.png'],
        placeholder='A conversation with VideoMind',
        label='VideoMind')

    prompt = gr.Textbox(label='Text Prompt', placeholder='Ask a question about the video...')

    with gr.Blocks(title=TITLE, js=JS) as demo:
        gr.HTML(LOGO)
        gr.HTML(BADGE)
        gr.Markdown(DISC)

        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Video()

                with gr.Group():
                    role = gr.CheckboxGroup(
                        choices=[('🗺️ Planner', 'pla'), ('🔍 Grounder', 'gnd'), ('📊 Verifier', 'ver'),
                                 ('📝 Answerer', 'ans')],
                        value=['pla', 'gnd', 'ver', 'ans'],
                        interactive=True,
                        label='Roles',
                        info='Select the role(s) you would like to activate.')
                    role.change(update_placeholder, role, prompt)

                    with gr.Accordion(label='Hyperparameters', open=False):
                        max_candidates = gr.Slider(
                            1,
                            100,
                            value=5,
                            step=1,
                            interactive=True,
                            label='Max Candidate Moments',
                            info='The maximum number of candidate moments in Grounder (Default: 5)')
                        temperature = gr.Slider(
                            0,
                            1,
                            value=0,
                            step=0.1,
                            interactive=True,
                            label='Temperature',
                            info='Higher value leads to more creativity and randomness (Default: 0)')
                        max_new_tokens = gr.Slider(
                            1,
                            1024,
                            value=256,
                            step=1,
                            interactive=True,
                            label='Max Output Tokens',
                            info='The maximum number of output tokens for each role (Default: 256)')

                prompt.render()

                with gr.Row():
                    random_btn = gr.Button(value='🔮 Random')
                    random_btn.click(random_sample, None, [video, prompt, role])

                    reset_btn = gr.ClearButton([video, prompt, chat], value='🗑️ Reset')
                    reset_btn.click(reset_components, None, [role, max_candidates, temperature, max_new_tokens])

                    submit_btn = gr.Button(value='🚀 Submit', variant='primary')
                    ctx = submit_btn.click(disable_btns, None, [random_btn, reset_btn, submit_btn])
                    ctx = ctx.then(main, [video, prompt, role, max_candidates, temperature, max_new_tokens], chat)
                    ctx.then(enable_btns, None, [random_btn, reset_btn, submit_btn])

                gr.Examples(examples=EXAMPLES, inputs=[video, prompt, role], examples_per_page=3)

            with gr.Column(scale=5):
                chat.render()

    return demo


if __name__ == '__main__':
    demo = build_demo()

    demo.queue()
    demo.launch(server_name='0.0.0.0', allowed_paths=[f'{PATH}/assets', f'{PATH}/examples'])
