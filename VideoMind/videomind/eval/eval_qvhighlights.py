# Modified from https://github.com/showlab/UniVTG/blob/main/eval/eval.py

import argparse
import copy
from collections import OrderedDict, defaultdict

import nncore
import numpy as np

from sklearn.metrics import precision_recall_curve


def compute_temporal_iou_batch_paired(a, b):
    intersection = np.maximum(0, np.minimum(a[:, 1], b[:, 1]) - np.maximum(a[:, 0], b[:, 0]))
    union = np.maximum(a[:, 1], b[:, 1]) - np.minimum(a[:, 0], b[:, 0])
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def compute_temporal_iou_batch_cross(spans1, spans2):
    areas1 = spans1[:, 1] - spans1[:, 0]
    areas2 = spans2[:, 1] - spans2[:, 0]
    l = np.maximum(spans1[:, None, 0], spans2[None, :, 0])
    r = np.minimum(spans1[:, None, 1], spans2[None, :, 1])
    inter = np.clip(r - l, 0, None)
    union = areas1[:, None] + areas2[None, :] - inter
    iou = inter / union
    return iou, union


def interpolated_precision_recall(prc, rec):
    mprc = np.hstack([[0], prc, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprc) - 1)[::-1]:
        mprc[i] = max(mprc[i], mprc[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprc[idx])
    return ap


def compute_average_precision_detection(annos, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    num_thresholds = len(tiou_thresholds)
    num_gts = len(annos)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    prediction.sort(key=lambda x: -x['score'])
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    ground_truth_by_videoid = dict()
    for i, item in enumerate(annos):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array([[pred['t-start'], pred['t-end']]])
        _gt = np.array([[gt['t-start'], gt['t-end']] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :], recall_cumsum[t_idx, :])

    return ap


def get_ap(y_true, y_pred, interpolate=True, point_11=False):
    assert len(y_true) == len(y_pred), 'Prediction and ground truth need to be of the same length'
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [0, 1], 'Ground truth can only contain elements {0,1}'

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    recall = recall.astype(np.float32)

    if interpolate:
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:
        precision_11 = [precision[np.where(recall >= t)[0][-1]] for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])


def compute_average_precision_detection_wrapper(input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, annos, prediction = input_triple
    scores = compute_average_precision_detection(annos, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_mr_ap(preds, annos, iou_thds=np.linspace(0.5, 0.95, 10), max_gt_windows=None, max_pred_windows=10):
    iou_thds = [float(f'{e:.2f}') for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in preds:
        pred_windows = d['pred_relevant_windows'][:max_pred_windows] \
            if max_pred_windows is not None else d['pred_relevant_windows']
        qid = d['qid']
        for w in pred_windows:
            pred_qid2data[qid].append({'video-id': d['qid'], 't-start': w[0], 't-end': w[1], 'score': w[2]})

    gt_qid2data = defaultdict(list)
    for d in annos:
        gt_windows = d['relevant_windows'][:max_gt_windows] \
            if max_gt_windows is not None else d['relevant_windows']
        qid = d['qid']
        for w in gt_windows:
            gt_qid2data[qid].append({'video-id': d['qid'], 't-start': w[0], 't-end': w[1]})
    qid2ap_list = dict()
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    for data_triple in data_triples:
        qid, scores = compute_ap_from_triple(data_triple)
        qid2ap_list[qid] = scores

    ap_array = np.array(list(qid2ap_list.values()))
    ap_thds = ap_array.mean(0)
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap['average'] = np.mean(ap_thds)

    iou_thd2ap = {k: float(f'{100 * v:.2f}') for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(preds, annos, iou_thds=np.linspace(0.3, 0.95, 14)):
    iou_thds = [float(f'{e:.2f}') for e in iou_thds]
    pred_qid2window = {d['qid']: d['pred_relevant_windows'][0][:2] for d in preds}
    gt_qid2window = dict()
    for d in annos:
        cur_gt_windows = d['relevant_windows']
        cur_qid = d['qid']
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d['relevant_windows']))[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = dict()
    miou_at_one = float(f'{np.mean(pred_gt_iou) * 100:.2f}')
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f'{np.mean(pred_gt_iou >= thd) * 100:.2f}')
    return iou_thd2recall_at_one, miou_at_one


def compute_mr_r5(preds, annos, iou_thds=np.linspace(0.3, 0.95, 14)):
    iou_thds = [float(f'{e:.2f}') for e in iou_thds]
    pred_qid2window = {d['qid']: [x[:2] for x in d['pred_relevant_windows'][:5]] for d in preds}
    gt_qid2window = dict()
    pred_optimal_qid2window = dict()
    for d in annos:
        cur_gt_windows = d['relevant_windows']
        cur_qid = d['qid']
        cur_max_iou_pred = 0
        cur_max_iou_gt = 0
        if len(cur_gt_windows) > 0:
            cur_ious = compute_temporal_iou_batch_cross(
                np.array(pred_qid2window[cur_qid]), np.array(d['relevant_windows']))[0]
            cur_ious[np.isnan(cur_ious)] = 0
            cur_max_iou_pred, cur_max_iou_gt = np.where(cur_ious == np.max(cur_ious))
            cur_max_iou_pred, cur_max_iou_gt = cur_max_iou_pred[0], cur_max_iou_gt[0]
        pred_optimal_qid2window[cur_qid] = pred_qid2window[cur_qid][cur_max_iou_pred]
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_gt]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_optimal_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = dict()
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f'{np.mean(pred_gt_iou >= thd) * 100:.2f}')
    return iou_thd2recall_at_one


def get_data_by_range(preds, annos, len_range):
    min_l, max_l = len_range
    if min_l == 0 and max_l == float('inf'):
        return preds, annos

    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in annos:
        rel_windows_in_range = [w for w in d['relevant_windows'] if min_l < (w[1] - w[0]) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d['relevant_windows'] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d['qid'])

    submission_in_range = []
    for d in preds:
        if d['qid'] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    if submission_in_range == ground_truth_in_range == []:
        return preds, annos

    return submission_in_range, ground_truth_in_range


def eval_moment_retrieval(preds, annos):
    length_ranges = [[0, 10], [10, 30], [30, float('inf')], [0, float('inf')]]
    range_names = ['short', 'middle', 'long', 'full']

    ret_metrics = dict()
    for l_range, name in zip(length_ranges, range_names):
        _submission, _ground_truth = get_data_by_range(preds, annos, l_range)
        print(f'{name}: {l_range}, {len(_ground_truth)}/{len(annos)}={100*len(_ground_truth)/len(annos):.2f} samples')
        iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth)
        iou_thd2recall_at_one, miou_at_one = compute_mr_r1(_submission, _ground_truth)
        iou_thd2recall_at_five = compute_mr_r5(_submission, _ground_truth)
        ret_metrics[name] = {
            'MR-mIoU': miou_at_one,
            'MR-mAP': iou_thd2average_precision,
            'MR-R1': iou_thd2recall_at_one,
            'MR-R5': iou_thd2recall_at_five
        }

    return ret_metrics


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):
    qid2max_scored_clip_idx = {k: np.argmax(v['pred_saliency_scores']) for k, v in qid2preds.items()}
    hit_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid]
        gt_scores_binary = qid2gt_scores_binary[qid]
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx]
    hit_at_one = float(f'{100 * np.mean(np.max(hit_scores, 1)):.2f}')
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary):
    qid2pred_scores = {k: v['pred_saliency_scores'] for k, v in qid2preds.items()}
    ap_scores = np.zeros((len(qid2preds), 3))
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        for w_idx in range(3):
            y_true = qid2gt_scores_binary[qid][:, w_idx]
            y_pred = np.array(qid2pred_scores[qid])
            input_tuples.append((idx, w_idx, y_true, y_pred))

    for input_tuple in input_tuples:
        idx, w_idx, score = compute_ap_from_tuple(input_tuple)
        ap_scores[idx, w_idx] = score

    mean_ap = float(f'{100 * np.mean(ap_scores):.2f}')
    return mean_ap


def compute_ap_from_tuple(input_tuple):
    idx, w_idx, y_true, y_pred = input_tuple
    if len(y_true) < len(y_pred):
        y_pred = y_pred[:len(y_true)]
    elif len(y_true) > len(y_pred):
        _y_predict = np.zeros(len(y_true))
        _y_predict[:len(y_pred)] = y_pred
        y_pred = _y_predict

    score = get_ap(y_true, y_pred)
    return idx, w_idx, score


def mk_gt_scores(gt_data, clip_length=2):
    num_clips = int(gt_data['duration'] / clip_length)
    saliency_scores_full_video = np.zeros((num_clips, 3))
    relevant_clip_ids = np.array(gt_data['relevant_clip_ids'])
    saliency_scores_relevant_clips = np.array(gt_data['saliency_scores'])
    saliency_scores_full_video[relevant_clip_ids] = saliency_scores_relevant_clips
    return saliency_scores_full_video


def eval_highlight(preds, annos):
    qid2preds = {d['qid']: d for d in preds}
    qid2gt_scores_full_range = {d['qid']: mk_gt_scores(d) for d in annos}
    gt_saliency_score_min_list = [2, 3, 4]
    saliency_score_names = ['Fair', 'Good', 'VeryGood']
    highlight_det_metrics = dict()
    for gt_saliency_score_min, score_name in zip(gt_saliency_score_min_list, saliency_score_names):
        qid2gt_scores_binary = {
            k: (v >= gt_saliency_score_min).astype(float)
            for k, v in qid2gt_scores_full_range.items()
        }
        hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
        mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
        highlight_det_metrics[f'HL-min-{score_name}'] = {'HL-mAP': mean_ap, 'HL-Hit1': hit_at_one}
    return highlight_det_metrics


def qvhighlights_eval(preds, annos):
    pred_qids = set([e['qid'] for e in preds])
    gt_qids = set([e['qid'] for e in annos])
    assert pred_qids == gt_qids, 'qids in annos and preds must match'

    eval_metrics = dict()
    eval_metrics_brief = OrderedDict()
    if 'pred_relevant_windows' in preds[0]:
        moment_ret_scores = eval_moment_retrieval(preds, annos)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            'MR-full-mAP': moment_ret_scores['full']['MR-mAP']['average'],
            'MR-full-mAP@0.5': moment_ret_scores['full']['MR-mAP']['0.5'],
            'MR-full-mAP@0.75': moment_ret_scores['full']['MR-mAP']['0.75'],
            'MR-short-mAP': moment_ret_scores['short']['MR-mAP']['average'],
            'MR-middle-mAP': moment_ret_scores['middle']['MR-mAP']['average'],
            'MR-long-mAP': moment_ret_scores['long']['MR-mAP']['average'],
            'MR-short-mIoU': moment_ret_scores['short']['MR-mIoU'],
            'MR-middle-mIoU': moment_ret_scores['middle']['MR-mIoU'],
            'MR-long-mIoU': moment_ret_scores['long']['MR-mIoU'],
            'MR-full-mIoU': moment_ret_scores['full']['MR-mIoU'],
            'MR-full-R1@0.3': moment_ret_scores['full']['MR-R1']['0.3'],
            'MR-full-R1@0.5': moment_ret_scores['full']['MR-R1']['0.5'],
            'MR-full-R1@0.7': moment_ret_scores['full']['MR-R1']['0.7'],
            'MR-full-R5@0.3': moment_ret_scores['full']['MR-R5']['0.3'],
            'MR-full-R5@0.5': moment_ret_scores['full']['MR-R5']['0.5'],
            'MR-full-R5@0.7': moment_ret_scores['full']['MR-R5']['0.7']
        }
        eval_metrics_brief.update(sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))

    if ('pred_saliency_scores' in preds[0]) and ('saliency_scores' in annos[0]):
        if isinstance(annos[0]['saliency_scores'], list):
            highlight_det_scores = eval_highlight(preds, annos)
            eval_metrics.update(highlight_det_scores)
            highlight_det_scores_brief = dict([(f"{k}-{sub_k.split('-')[1]}", v[sub_k])
                                               for k, v in highlight_det_scores.items() for sub_k in v])
            eval_metrics_brief.update(highlight_det_scores_brief)
            eval_metrics_brief['HL-min-VeryGood-mAP'] = eval_metrics_brief.pop('HL-min-VeryGood-mAP')
            eval_metrics_brief['HL-min-VeryGood-Hit1'] = eval_metrics_brief.pop('HL-min-VeryGood-Hit1')

    final_eval_metrics = OrderedDict()
    final_eval_metrics['brief'] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    parser.add_argument('--anno_path', default='data/qvhighlights/highlight_val_release.jsonl')
    parser.add_argument('--out_name', default='metrics.log')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if nncore.is_dir(args.pred_path):
        log_file = nncore.join(args.pred_path, args.out_name)
    else:
        log_file = nncore.same_dir(args.pred_path, args.out_name)

    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    if nncore.is_dir(args.pred_path):
        pred_paths = nncore.ls(args.pred_path, ext=['json', 'jsonl'], join_path=True, sort=True)
        nncore.log(f'Total number of files: {len(pred_paths)}\n')
        preds = nncore.flatten([nncore.load(p) for p in pred_paths])
    else:
        nncore.log(f'Loading predictions from {args.pred_path}')
        preds = nncore.load(args.pred_path)

    annos = nncore.load(args.anno_path)

    res = qvhighlights_eval(preds, annos)['brief']
    for k, v in res.items():
        nncore.log(f'{k}: {v}')
