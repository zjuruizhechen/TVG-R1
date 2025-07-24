# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import re


def parse_span(span, duration, min_len=-1):
    s, e = span
    s, e = min(duration, max(0, s)), min(duration, max(0, e))
    s, e = min(s, e), max(s, e)

    if min_len != -1 and e - s < min_len:
        h = min_len / 2
        c = min(duration - h, max(h, (s + e) / 2))
        s, e = c - h, c + h

    s, e = min(duration, max(0, s)), min(duration, max(0, e))
    return s, e


def parse_query(query):
    return re.sub(r'\s+', ' ', query).strip().strip('.').strip()


def parse_question(question):
    return re.sub(r'\s+', ' ', question).strip()
