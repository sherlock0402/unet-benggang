import os
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)

    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    dice = 2 * np.sum(intersection) / (np.sum(gt_mask) + np.sum(pred_mask) + 1e-6)

    precision = precision_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
    recall = recall_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
    f1 = f1_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)

    return iou, dice, precision, recall, f1


def evaluate_metrics(gt_folder, pred_folder):
    iou_list, dice_list, precision_list, recall_list, f1_list = [], [], [], [], []

    pred_files = sorted(os.listdir(pred_folder))  # 以预测文件为基准
    for file in pred_files:
        gt_path = os.path.join(gt_folder, file)
        pred_path = os.path.join(pred_folder, file)

        if not os.path.exists(gt_path):
            print(f"Warning: {file} not found in ground truth folder.")
            continue

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        iou, dice, precision, recall, f1 = compute_metrics(gt_mask, pred_mask)
        iou_list.append(iou)
        dice_list.append(dice)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "mDice": np.mean(dice_list),
        "mPrecision": np.mean(precision_list),
        "mRecall": np.mean(recall_list),
        "mF1": np.mean(f1_list)
    }


# 示例：假设GT和预测结果在以下文件夹
gt_folder = "/data/code2025/Q1/2025-02-10-01/VOCdevkit/VOC2007/SegmentationClass"
pred_folder = "/data/code2025/Q1/2025-02-10-01/miou_out/unet_att/detection-results"

metrics = evaluate_metrics(gt_folder, pred_folder)
print(metrics)

# {'mDice': 0.8268987888871244, 'mPrecision': 0.8384631830029261, 'mRecall': 0.8435804566652236, 'mF1': 0.826898789079476}
# {'mDice': 0.8054178616862011, 'mPrecision': 0.7865279054034328, 'mRecall': 0.8557995680509318, 'mF1': 0.8054178618713765}
