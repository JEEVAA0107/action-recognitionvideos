import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import Config
from utils import set_seed, calculate_topk_accuracy
from transforms import get_video_transforms
from dataset import VideoActionDataset
from models import get_model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one training epoch over the dataset.
    """
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0

    for idx, (clips, labels) in enumerate(dataloader):
        clips, labels = clips.to(device), labels.to(device)
        batch_size = labels.size(0)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        top1, top5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
        running_top1 += top1 * batch_size
        running_top5 += top5 * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_top1 = running_top1 / total_samples
    epoch_top5 = running_top5 / total_samples

    return epoch_loss, epoch_top1, epoch_top5

def validate(model, dataloader, criterion, device):
    """
    Executes validation evaluation.
    """
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0

    with torch.no_grad():
        for clips, labels in dataloader:
            clips, labels = clips.to(device), labels.to(device)
            batch_size = labels.size(0)

            outputs = model(clips)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            top1, top5 = calculate_topk_accuracy(outputs, labels, topk=(1, 5))
            running_top1 += top1 * batch_size
            running_top5 += top5 * batch_size
            total_samples += batch_size

    val_loss = running_loss / total_samples
    val_top1 = running_top1 / total_samples
    val_top5 = running_top5 / total_samples

    return val_loss, val_top1, val_top5

def run_training(args):
    """
    Main training execution function.
    """
    set_seed(Config.SEED)
    Config.setup_directories()

    device = torch.device(args.device)
    print(f"[INFO] Using Device: {device}")

    # Prepare Transforms
    train_transform = get_video_transforms(is_train=True, frame_size=(Config.FRAME_HEIGHT, Config.FRAME_WIDTH))
    val_transform = get_video_transforms(is_train=False, frame_size=(Config.FRAME_HEIGHT, Config.FRAME_WIDTH))

    # Initialize Datasets
    train_dataset = VideoActionDataset(root_dir=args.train_dir, transform=train_transform)
    val_dataset = VideoActionDataset(root_dir=args.val_dir, transform=val_transform, class_to_idx=train_dataset.class_to_idx)

    if len(train_dataset) == 0:
        print("[ERROR] Training dataset is empty! Run 'python create_sample_dataset.py' to generate sample data.")
        return

    print(f"[INFO] Action Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"[INFO] Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)

    # Initialize Model
    model = get_model(num_classes=len(train_dataset.classes), model_name=args.model_name, pretrained=args.pretrained).to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_top1 = 0.0

    print("\n" + "="*60)
    print(f"       STARTING TRAINING: {args.model_name.upper()}")
    print("="*60)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        
        train_loss, train_top1, train_top5 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        ep_time = time.time() - ep_start

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Top-1: {train_top1:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%")

        # Save Best Model Checkpoint
        if val_top1 >= best_val_top1:
            best_val_top1 = val_top1
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_top1': best_val_top1,
                'class_to_idx': train_dataset.class_to_idx,
                'classes': train_dataset.classes,
                'model_name': args.model_name
            }
            torch.save(checkpoint, Config.BEST_MODEL_PATH)
            print(f"  [SAVED] Best model saved to: {Config.BEST_MODEL_PATH}")

    total_time = time.time() - start_time
    print("="*60)
    print(f"[COMPLETED] Training finished in {total_time/60:.2f} mins. Best Validation Top-1: {best_val_top1:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Action Recognition Model")
    parser.add_argument("--train_dir", type=str, default=Config.TRAIN_DIR, help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default=Config.VAL_DIR, help="Path to validation data directory")
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--model_name", type=str, default=Config.MODEL_NAME, choices=['r2plus1d_18', 'r3d_18', 'mc3_18'], help="Model architecture")
    parser.add_argument("--pretrained", type=bool, default=Config.PRETRAINED, help="Use pretrained backbone weights")
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="Device (cuda or cpu)")

    args = parser.parse_args()
    run_training(args)
