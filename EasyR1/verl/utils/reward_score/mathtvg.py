import re
import json

from mathruler.grader import extract_boxed_content, grade_answer

def temporal_iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    _iou=max(min1 - max0, 0) / (max1 - min0)
    return max(0,_iou)

def tvg_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<time>.*</time>.*<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def extract_time_content(predict_str: str) -> str:
    pattern = r'<time>(.*?)</time>'
    
    match = re.search(pattern, predict_str)
    
    if match:
        return match.group(1)
    
    return ""

def tvg_accuracy_reward(predict_str: str, gt_frame: list) -> float:
    try:
        predict_time = extract_time_content(predict_str)

        for number in gt_frame:
            if abs(predict_time - number) < 30:
                reward = 1

        return reward
    except Exception as e:
        # print(e)
        pass  # Continue to next verification method if this fails

    return 0.0

def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def mathtvg_compute_score(predict_str: str, ground_truth: list) -> float:
    # print(predict_str, ground_truth, video_length)
    acc_reward = math_acc_reward(predict_str, ground_truth)
    format_reward = tvg_format_reward(predict_str)
    # print(f"acc: {acc_reward}, format: {format_reward}")
    reward = 0.9 * acc_reward + 0.1 * format_reward
    # reward /= 2
    return reward
