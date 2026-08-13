import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from utils import set_seed, calculate_topk_accuracy, plot_confusion_matrix, generate_classification_report
from transforms import get_video_transforms
from dataset import VideoActionDataset
from models import get_model

def evaluate_model(args):
    """
    Evaluates trained action recognition model on test dataset and generates metrics.
    """
    set_seed(Config.SEED)

    device = torch.device(args.device)
    print(f"[INFO] Evaluating model on device: {device}")

    # Check for saved model checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"[ERROR] Checkpoint file not found at: {args.checkpoint_path}")
        print("[HINT] Run 'python train.py' first to train and save model weights.")
        return

    # Load Checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    classes = checkpoint.get('classes', Config.CLASSES)
    class_to_idx = checkpoint.get('class_to_idx', {cls: i for i, cls in enumerate(classes)})
    model_name = checkpoint.get('model_name', Config.MODEL_NAME)

    print(f"[INFO] Loaded checkpoint from Epoch {checkpoint['epoch']} (Best Val Top-1: {checkpoint['best_val_top1']:.2f}%)")
    print(f"[INFO] Model Architecture: {model_name} | Action Classes: {classes}")

    # Initialize Model and load weights
    model = get_model(num_classes=len(classes), model_name=model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Dataset & Dataloader
    test_transform = get_video_transforms(is_train=False, frame_size=(Config.FRAME_HEIGHT, Config.FRAME_WIDTH))
    test_dataset = VideoActionDataset(root_dir=args.test_dir, transform=test_transform, class_to_idx=class_to_idx)

    if len(test_dataset) == 0:
        print(f"[ERROR] Test directory '{args.test_dir}' is empty.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS)

    all_preds = []
    all_targets = []

    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0
    inference_times = []

    print("\n" + "="*60)
    print("                EVALUATION & BENCHMARKING")
    print("="*60)

    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)

            # Benchmark Inference Latency
            start_t = time.perf_counter()
            outputs = model(clips)
            end_t = time.perf_counter()

            latency_ms = (end_t - start_t) * 1000.0
            inference_times.append(latency_ms)

            _, preds = torch.max(outputs, 1)

            top1, top5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
            running_top1 += top1
            running_top5 += top5

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            total_samples += 1

    top1_acc = running_top1 / total_samples
    top5_acc = running_top5 / total_samples
    avg_latency = sum(inference_times) / len(inference_times)

    print(f"[RESULTS] Test Samples Evaluated: {total_samples}")
    print(f"[RESULTS] Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"[RESULTS] Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"[RESULTS] Average Inference Latency per clip: {avg_latency:.2f} ms")

    # Generate Detailed Classification Report
    report = generate_classification_report(all_targets, all_preds, class_names=classes)
    print("\n[PERFORMANCE METRICS REPORT]")
    print(report)

    # Plot and Save Confusion Matrix
    cm_output_path = os.path.join(Config.OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(all_targets, all_preds, class_names=classes, output_path=cm_output_path)

    try:
        if os.name == 'nt' and os.path.exists(cm_output_path):
            os.startfile(cm_output_path)
            print(f"[INFO] Opening confusion matrix image...")
    except Exception:
        pass

    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Action Recognition Model")
    parser.add_argument("--test_dir", type=str, default=Config.TEST_DIR, help="Path to test dataset directory")
    parser.add_argument("--checkpoint_path", type=str, default=Config.BEST_MODEL_PATH, help="Path to saved model checkpoint")
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="Device (cuda or cpu)")

    args = parser.parse_args()
    evaluate_model(args)
