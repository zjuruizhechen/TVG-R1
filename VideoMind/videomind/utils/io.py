# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import pysrt
from decord import VideoReader


def time_to_seconds(t):
    return (t.hour * 60 + t.minute) * 60 + t.second + t.microsecond / 1000000


def load_subtitle(path):
    subs = pysrt.open(path)

    parsed = []
    for sub in subs:
        s = time_to_seconds(sub.start.to_time())
        e = time_to_seconds(sub.end.to_time())
        parsed.append((s, e, sub.text))

    return parsed


def get_duration(path, num_threads=1):
    # sometimes the video is loaded as a list of frames
    if isinstance(path, list):
        return len(path)

    vr = VideoReader(path, num_threads=num_threads)
    duration = len(vr) / vr.get_avg_fps()
    return duration
