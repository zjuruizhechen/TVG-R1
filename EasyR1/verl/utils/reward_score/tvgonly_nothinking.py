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
    pattern = re.compile(r"<time>.*</time>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def extract_time_content(predict_str: str) -> str:
    pattern = r'<time>(.*?)</time>'
    
    match = re.search(pattern, predict_str)
    
    if match:
        return match.group(1)
    
    return ""

def tvg_accuracy_reward(predict_str: str, ground_truth: list, video_length: float) -> float:
    try:
        content_answer_match = re.search(r"<time>(.*?)</time>", predict_str, re.DOTALL)
        content_answer = content_answer_match.group(1).strip()
        start_time, end_time = json.loads(content_answer)

        # 时间归一化到 [0, 1]
        answer_timestamp = [
            float(start_time) / video_length,
            float(end_time) / video_length
        ]
        ground_truth_timestamp = [
            float(ground_truth[0]) / video_length,
            float(ground_truth[1]) / video_length
        ]
        reward = temporal_iou(answer_timestamp, ground_truth_timestamp)

        print(f"[pred]: {answer_timestamp} [gt]: {ground_truth_timestamp} [reward]: {reward}")
        # print(f"gt: {ground_truth}, pred: {answer_timestamp}")

        # reward = 1.0 if reward >= 0.5 else 0
        return reward
    except Exception as e:
        print(predict_str)
        # print(e)
        pass  # Continue to next verification method if this fails

    return 0.0


def math_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def tvgonly_nothinking_compute_score(predict_str: str, ground_truth: list, gt_frame: list, video_length: float) -> float:
    # print(predict_str, ground_truth, video_length)
    tvg_reward = tvg_accuracy_reward(predict_str, gt_frame, video_length)
    format_reward = tvg_format_reward(predict_str)
    # print(f"acc: {acc_reward}, format: {format_reward}")
    reward = 0.1 * format_reward + 0.9 * tvg_reward
    # reward /= 2
    return reward
