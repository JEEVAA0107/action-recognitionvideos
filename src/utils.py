import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def set_seed(seed=42):
    """
    Set fixed random seeds for Python, NumPy, and PyTorch to ensure 100% reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global random seed fixed to: {seed}")

def calculate_topk_accuracy(output, target, topk=(1, 5)):
    """
    Computes top-k accuracy for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Handle maxk greater than number of classes
        num_classes = output.size(1)
        maxk = min(maxk, num_classes)
        actual_topk = tuple(min(k, num_classes) for k in topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in actual_topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

        # If top-5 was requested but num_classes < 5, pad output to maintain tuple length 2
        while len(res) < len(topk):
            res.append(res[0])
            
        return res

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Generates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Action Recognition Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {output_path}")

def generate_classification_report(y_true, y_pred, class_names):
    """
    Generates a detailed text classification report (Precision, Recall, F1-Score).
    """
    report = classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, output_dict=False, zero_division=0)
    return report
