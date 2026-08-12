import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config
from transforms import get_video_transforms

class VideoActionDataset(Dataset):
    """
    Dataset class for video action recognition.
    Expects dataset directory layout:
        root_dir/
            class_1/
                video1.mp4
                video2.avi
            class_2/
                video3.mp4
    """
    def __init__(self, root_dir, num_frames=Config.NUM_FRAMES, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        
        self.video_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            print(f"[WARNING] Directory {root_dir} does not exist yet.")
            self.classes = Config.CLASSES
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            return

        # Find all class folders
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        if class_to_idx is None:
            self.classes = classes if len(classes) > 0 else Config.CLASSES
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(self.class_to_idx.keys())

        # Collect video file paths
        valid_extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm")
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            cls_idx = self.class_to_idx[cls_name]
            
            for ext in valid_extensions:
                for video_path in glob.glob(os.path.join(cls_folder, ext)):
                    self.video_paths.append(video_path)
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.video_paths)

    def _sample_frame_indices(self, total_frames):
        """
        Uniformly sample self.num_frames frame indices across total_frames.
        """
        if total_frames <= 0:
            return [0] * self.num_frames
            
        if total_frames < self.num_frames:
            # Repeat frames if video duration is too short
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
        return indices

    def _read_video_frames(self, video_path):
        """
        Extract sampled RGB frames from video using OpenCV.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            # Read sequentially to count frames if metadata is missing
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            if len(frames) == 0:
                # Return blank frames fallback if unreadable
                return torch.zeros((self.num_frames, Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), dtype=torch.uint8)
                
            sample_indices = self._sample_frame_indices(len(frames))
            sampled_frames = [frames[idx] for idx in sample_indices]
            return torch.from_numpy(np.array(sampled_frames, dtype=np.uint8))

        sample_indices = set(self._sample_frame_indices(total_frames))
        sorted_indices = sorted(list(sample_indices))
        
        extracted_frames = {}
        current_frame_idx = 0
        
        while cap.isOpened() and len(extracted_frames) < len(sorted_indices):
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame_idx in sample_indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted_frames[current_frame_idx] = frame_rgb
            current_frame_idx += 1
            
        cap.release()
        
        # Build ordered frame sequence
        ordered_indices = self._sample_frame_indices(total_frames)
        pad_shape = next(iter(extracted_frames.values())).shape if len(extracted_frames) > 0 else (Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3)
        final_frames = []
        
        for idx in ordered_indices:
            frame = extracted_frames.get(idx, np.zeros(pad_shape, dtype=np.uint8))
            final_frames.append(frame)

        return torch.from_numpy(np.array(final_frames, dtype=np.uint8))

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract frames as Tensor (T, H, W, C)
        frames_tensor = self._read_video_frames(video_path)

        # Apply transforms -> Output shape (C, T, H, W)
        if self.transform:
            clip_tensor = self.transform(frames_tensor)
        else:
            clip_tensor = frames_tensor.permute(3, 0, 1, 2).float() / 255.0

        return clip_tensor, label


if __name__ == "__main__":
    # Smoke test for Dataset class
    print("[TEST] Initializing VideoActionDataset...")
    transform = get_video_transforms(is_train=True)
    ds = VideoActionDataset(root_dir=Config.TRAIN_DIR, transform=transform)
    print(f"[TEST] Dataset loaded. Number of samples: {len(ds)}, Classes: {ds.classes}")
