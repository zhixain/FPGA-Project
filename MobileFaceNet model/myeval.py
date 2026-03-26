import argparse
import functools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import torch
from utils.utils import (
    add_arguments, print_arguments, get_lfw_list,
    get_features, get_feature_dict, test_performance
)


def get_confusion_counts(feature_dict, lfw_data_list, threshold):
    with open(lfw_data_list, 'r') as fd:
        pairs = fd.readlines()

    TP = TN = FP = FN = 0
    for pair in pairs:
        splits = pair.split()
        f1 = feature_dict[splits[0]]
        f2 = feature_dict[splits[1]]
        label = int(splits[2])
        sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        pred = 1 if sim >= threshold else 0

        if label == 1 and pred == 1:
            TP += 1
        elif label == 1 and pred == 0:
            FN += 1
        elif label == 0 and pred == 0:
            TN += 1
        elif label == 0 and pred == 1:
            FP += 1

    return TN, FP, FN, TP


def eval(args, model, label='model'):
    img_paths = get_lfw_list(args.test_list_path)
    features = get_features(model, img_paths, batch_size=args.batch_size)
    fe_dict = get_feature_dict(img_paths, features)

    accuracy, threshold = test_performance(fe_dict, args.test_list_path)
    TN, FP, FN, TP = get_confusion_counts(fe_dict, args.test_list_path, threshold)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"[{label}] 準確度 (Accuracy): {accuracy:.4f}")
    print(f"[{label}] Precision: {precision:.4f}")
    print(f"[{label}] Recall:    {recall:.4f}")
    print(f"[{label}] F1-score:  {f1_score:.4f}")

    y_score, y_true = [], []
    with open(args.test_list_path, 'r') as fd:
        pairs = fd.readlines()
        for pair in pairs:
            splits = pair.strip().split()
            f1 = fe_dict[splits[0]]
            f2 = fe_dict[splits[1]]
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            y_score.append(sim)
            y_true.append(int(splits[2]))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, label


def plot_roc_curve(model_results):
    plt.figure(figsize=(8, 6))
    for fpr, tpr, roc_auc, label in model_results:
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    model_paths = [
        ('save_model_128/mobilefacenet.pth', 'MobileFaceNet_128'),
        ('save_model_256/mobilefacenet.pth', 'MobileFaceNet_256'),
        ('save_model_512/mobilefacenet.pth', 'MobileFaceNet_512'),
        ('mobilefacenet.pth', 'MobileFaceNet_512_author')
    ]

    model_results = []

    for path, label in model_paths:
        print(f"\n=== 測試模型: {label} ===")
        args.model_path = path
        model = torch.load(args.model_path)
        model.to(torch.device("cuda"))
        model.eval()

        fpr, tpr, roc_auc, label = eval(args, model, label)
        model_results.append((fpr, tpr, roc_auc, label))

    plot_roc_curve(model_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate face recognition models and plot ROC curves')
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('batch_size',       int,    64,                                 '测试批量大小')
    add_arg('test_list_path',   str,    'dataset/lfw_test.txt',             '测试数据路径')
    add_arg('model_path',       str,    'save_model/mobilefacenet.pth',     '模型路径')
    args = parser.parse_args()

    print_arguments(args)
    main()
