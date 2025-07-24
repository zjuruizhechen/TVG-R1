# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

IGNORE_INDEX = -100

REG_TOKEN = '<|reg|>'

SEG_S_TOKEN = '<|seg_start|>'
SEG_E_TOKEN = '<|seg_end|>'

PLANNER_PROMPT = (
    'You are acting as the planner now. '
    'Given a question about the video, your task is to analyze the question and identify the best way to answer this question. '
    'You have access to the following tools:\n\n'
    'Grounder: Accepts a text query and localize the relevant video segment according to the query.\n'
    'Verifier: A tool supporting grounder by verifying the reliability of its outputs.\n'
    'Answerer: Answer a given question directly based on the whole video or a cropped video segment.\n\n'
    'Your response must be a list in JSON format. '
    'A valid plan for reasoning could be "grounder, verifier, answer", "grounder, verifier", or "answerer", depending on the given question. '
    'Please see an example for the format below.\n\n'
    '[{{"type": "grounder", "value": "<text query>"}}, {{"type": "verifier"}}, {{"type": "answerer"}}]\n\n'
    'Note that only the grounder can accept an argument called "value", which is the text query used for grounding. '
    "Now I give you the question: '{}'. "
    'Please think carefully and respond with your plan in JSON directly.')

GROUNDER_PROMPT = (
    "You FIRST think about the reasoning process in the mind and finally determine the precise time period related to the query. "
  "The reasoning process MUST BE enclosed within <think> </think> tags. The specific time period MUST BE in the format [start time, end time] in seconds enclosed within <time> </time> tags. For example, <think>the reasoning process</think> <time>[5.2, 10.4]</time>. "
    "Now I give you the query: '{}'. ")

TIMER1_TEMPLATE = """To accurately pinpoint the event "[EVENT]" in the video, determine the precise time period of the event.
Output your thought process within the <think> </think> tags, including analysis with either specific timestamps (xx.xx) or time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. For example: "12.54 to 17.83"."""


GROUNDING_PROMPT = (
    "You FIRST think about the reasoning process in the mind and finally determine the precise time period related to the query. "
  "The reasoning process MUST BE enclosed within <think> </think> tags. The specific time period MUST BE in the format [start time, end time] in seconds enclosed within <time> </time> tags. For example, <think>the reasoning process</think> <time>[5.2, 10.4]</time>. ")

GROUNDEDANSWER_PROMPT = (
    """You FIRST think about the reasoning process in the mind and finally provide the final option answer. The reasoning process MUST include analysis with the specific time period in the video.
  The reasoning process MUST BE enclosed within <think> </think> tags. The specific time period MUST BE in the format [start time, end time] in seconds enclosed within <time> </time> tags. The final option answer MUST BE put in \\boxed{}. For example, <think>the reasoning process</think> <time>[5.2, 10.4]</time> \\boxed{C}. """)

ANSWER_PROMPT = (
    """You FIRST think about the reasoning process in the mind and finally provide the final option answer.
  The reasoning process MUST BE enclosed within <think> </think> tags. The final option answer MUST BE put in \\boxed{}. For example, <think>the reasoning process</think>\\boxed{C}. """)

VERIFIER_PROMPT = (
    'You are acting as the verifier now. '
    'You will be presented a text query describing a moment that potentialy happens in the given video. '
    f'Your task is to identify whether the video segment between {SEG_S_TOKEN} and {SEG_E_TOKEN} perfectly covers the moment. '
    f'If the described moment can be seen in the video, please focus on verifying whether the moment starts at {SEG_S_TOKEN} and ends at {SEG_E_TOKEN}. '
    "Respond with 'Yes' if you think the moment boundaries are correct, otherwise 'No'. "
    "If the described moment cannot be seen in the video, respond with 'No' directly. "
    "Now I give you the query: '{}'. "
    "Please think carefully and respond with 'Yes' or 'No' directly.")


VideoR1_QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)
