# Modified from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

import base64
import math
import warnings
from io import BytesIO

import decord
import numpy as np
import torch
from PIL import Image, ImageSequence
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import requests
from videomind.constants import IGNORE_INDEX
from videomind.conversation import get_conv

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 32 * 28 * 28
VIDEO_MAX_PIXELS = 128 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 128


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    # change order here to ensure not exceeding max_pixels
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        nframes = min(nframes, total_frames)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_gif(path):
    gif = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frames.append(np.array(frame.convert('RGB')))
    frames = np.stack(frames, axis=0)
    return frames


def _read_video_decord(ele: dict, ) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if video_path.endswith('.gif'):
        video = _read_video_gif(video_path)
        total_frames, video_fps = video.shape[0], ele.get('fps', FPS)
    else:
        vr = decord.VideoReader(video_path, num_threads=ele.get('num_threads', 0))
        total_frames, video_fps = len(vr), vr.get_avg_fps()

    # 1. re-calculate total frames
    s = ele.get('video_start')
    s = 0 if s is None else s
    e = ele.get('video_end')
    e = total_frames / video_fps if e is None else e
    s_frame = min(max(0, round(s * video_fps)), total_frames - 1)
    e_frame = min(max(0, round(e * video_fps)), total_frames - 1)
    if s_frame > e_frame:
        warnings.warn(f's_frame ({s_frame}) is greater than e_frame ({e_frame}), total_frames: {total_frames}')
        s_frame, e_frame = e_frame, s_frame

    # TODO: the actual total_frames shall be computed by e_frame - s_frame + 1
    # but it would affect verifier's performance when video_start and video_end get clamped
    # shall be fixed by using normalized timestamps instead of real time
    total_frames = min(max(FPS_MIN_FRAMES, round((e - s) * video_fps)), total_frames)

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    # 2. generate frame ids
    idx = torch.linspace(s_frame, e_frame, nframes).round().long().tolist()
    assert len(idx) == nframes, (len(idx), nframes)

    if video_path.endswith('.gif'):
        video = video[idx]
    else:
        video = vr.get_batch(idx).asnumpy()

    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, sanity_check=False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video = _read_video_decord(ele)
        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        if sanity_check and (video == 0).all():
            raise ValueError("video '{}' contains all zeros".format(ele["video"]))

        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({
                "image": video_element,
                **process_info
            }, size_factor=image_factor) for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ("image" in ele or "image_url" in ele or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
        conversations: list[dict] | list[list[dict]],
        sanity_check=False) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    # Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info, sanity_check=sanity_check))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs


def preprocess_chatml(input_ids, text, tokenizer):
    conv = get_conv('chatml')

    rounds = [m + conv.seps[0] for m in text.split(conv.seps[0])]
    assert (len(rounds) % 2 == 0) == (conv.system is not None)
    assert rounds[-1] == conv.seps[0]
    rounds = rounds[:-1]

    if conv.system is None:
        rounds = [''.join(rounds[i:i + 2]) for i in range(0, len(rounds), 2)]
    else:
        rounds = [''.join(rounds[:3])] + [''.join(rounds[i:i + 2]) for i in range(3, len(rounds), 2)]

    labels = input_ids.clone()

    sep = conv.seps[0] + conv.roles[1]
    cur_len = 0

    for i, rou in enumerate(rounds):
        if len(rou) == 0:
            break

        ins = sep.join(rou.split(sep)[:-1]) + sep

        rou_len = tokenizer(rou, return_length=True).length[0]
        ins_len = tokenizer(ins, return_length=True).length[0]

        labels[cur_len:cur_len + ins_len] = IGNORE_INDEX
        cur_len += rou_len

    if labels.size(0) != cur_len:
        warnings.warn(f'Tokenization mismatch: {labels.size(0)} and {cur_len}')

    return labels


def preprocess(input_ids, text, tokenizer, conv_type):
    if conv_type == 'chatml':
        return preprocess_chatml(input_ids, text, tokenizer)
    else:
        raise ValueError(f'unknown conversation type: {conv_type}')
