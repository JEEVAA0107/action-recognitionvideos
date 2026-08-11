import os
import urllib.request
import glob
import shutil
import random
import subprocess
import ssl
from config import Config
from utils import set_seed

# Bypass SSL verification to fix UCF101 certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

UCF101_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
# Using a subset of 10 classes to keep training time manageable on Colab
SAMPLE_CLASSES = [
    "Archery", "Basketball", "Biking", "Bowling", "BoxingPunching", 
    "Diving", "Fencing", "GolfSwing", "HorseRiding", "SoccerJuggling"
]

def download_file(url, dest):
    """Downloads a file from a URL with a simple progress print."""
    if os.path.exists(dest):
        print(f"[INFO] File {dest} already exists. Skipping download.")
        return

    print(f"[INFO] Downloading {url} to {dest}...")
    print("[INFO] This may take a while (approx. 6.5 GB). Please wait...")
    
    # Custom report hook for progress
    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            if block_num % 20000 == 0:  # Print occasionally
                print(f"Downloaded: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                
    try:
        urllib.request.urlretrieve(url, dest, reporthook=report)
        print(f"\n[SUCCESS] Downloaded to {dest}")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {url}. Error: {e}")

def extract_rar(archive_path, extract_dir):
    """Extracts a .rar file using the system 'unrar' command."""
    print(f"[INFO] Extracting {archive_path} to {extract_dir}...")
    try:
        # We assume 'unrar' is installed on the system (e.g., in Google Colab)
        subprocess.run(["unrar", "x", "-y", archive_path, extract_dir], check=True, stdout=subprocess.DEVNULL)
        print(f"[SUCCESS] Extraction complete.")
    except Exception as e:
        print(f"[ERROR] Failed to extract {archive_path}. Make sure 'unrar' is installed. Error: {e}")

def setup_ucf101_benchmark_structure(data_dir=Config.DATA_DIR, train_ratio=0.7, val_ratio=0.15):
    """
    Downloads, extracts, and structures the UCF101 dataset.
    """
    set_seed(Config.SEED)
    print("="*60)
    print("      AUTOMATED UCF101 DATASET DOWNLOAD & PREPARATION")
    print("="*60)
    
    os.makedirs(data_dir, exist_ok=True)
    rar_path = os.path.join(data_dir, "UCF101.rar")
    raw_extracted_dir = os.path.join(data_dir, "UCF-101")  # Default UCF101 extraction folder
    
    # Step 1: Download
    download_file(UCF101_URL, rar_path)
    
    # Step 2: Extract
    if not os.path.exists(raw_extracted_dir):
        extract_rar(rar_path, data_dir)
    else:
        print(f"[INFO] Folder {raw_extracted_dir} already exists. Skipping extraction.")
        
    # Step 3: Split and organize
    split_raw_dataset_folder(raw_extracted_dir, data_dir, train_ratio, val_ratio)

def split_raw_dataset_folder(raw_dir, target_data_dir=Config.DATA_DIR, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the extracted UCF101 folder into train/val/test directories for the selected classes.
    """
    if not os.path.exists(raw_dir):
        print(f"[ERROR] Raw directory '{raw_dir}' not found.")
        return

    print(f"[INFO] Filtering and splitting {len(SAMPLE_CLASSES)} selected classes...")

    for cls_name in SAMPLE_CLASSES:
        cls_raw_path = os.path.join(raw_dir, cls_name)
        if not os.path.exists(cls_raw_path):
            print(f"[WARNING] Class folder '{cls_name}' not found in raw dataset. Skipping.")
            continue
            
        video_files = glob.glob(os.path.join(cls_raw_path, "*.avi"))
        random.shuffle(video_files)
        total = len(video_files)
        
        if total == 0:
            print(f"[WARNING] No videos found for class '{cls_name}'.")
            continue
            
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        train_files = video_files[:train_count]
        val_files = video_files[train_count:train_count + val_count]
        test_files = video_files[train_count + val_count:]

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split_name, files in splits.items():
            dest_folder = os.path.join(target_data_dir, split_name, cls_name)
            os.makedirs(dest_folder, exist_ok=True)
            for fpath in files:
                # Copy file
                shutil.copy(fpath, os.path.join(dest_folder, os.path.basename(fpath)))

        print(f"  └─► Class '{cls_name}': {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test clips.")

    print("\n[SUCCESS] Dataset preparation complete! You can now run 'python train.py'.")

if __name__ == "__main__":
    setup_ucf101_benchmark_structure()
