import random
import torch
import torchvision.transforms.functional as F

class ComposeVideoTransforms:
    """
    Applies a sequence of transformations to a video clip tensor of shape (T, H, W, C) or (C, T, H, W).
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

class VideoResize:
    """
    Resizes each frame of a video clip tensor to specified (height, width).
    Expected clip tensor shape: (T, C, H, W) or (T, H, W, C) -> converted to (T, C, H, W)
    """
    def __init__(self, size):
        self.size = size  # (H, W)

    def __call__(self, clip):
        # Ensure clip format (T, C, H, W)
        if clip.dim() == 4 and clip.size(-1) in [1, 3]:  # (T, H, W, C)
            clip = clip.permute(0, 3, 1, 2)
        
        resized_frames = [F.resize(frame, self.size, antialias=True) for frame in clip]
        return torch.stack(resized_frames)

class VideoRandomHorizontalFlip:
    """
    Applies Random Horizontal Flip consistently across all frames in a video clip.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        if random.random() < self.p:
            flipped_frames = [F.hflip(frame) for frame in clip]
            return torch.stack(flipped_frames)
        return clip

class VideoNormalize:
    """
    Normalizes video clip tensor using mean and std.
    Input clip shape: (T, C, H, W) -> Returns: (C, T, H, W) normalized
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        # Convert clip from uint8 [0, 255] or float [0, 1]
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0

        normalized_frames = [F.normalize(frame, mean=self.mean, std=self.std) for frame in clip]
        stacked = torch.stack(normalized_frames)  # (T, C, H, W)
        # Permute to 3D CNN expected input format: (C, T, H, W)
        return stacked.permute(1, 0, 2, 3)

def get_video_transforms(is_train=True, frame_size=(112, 112)):
    """
    Builds spatiotemporal transformation pipeline for training and evaluation.
    """
    if is_train:
        return ComposeVideoTransforms([
            VideoResize(frame_size),
            VideoRandomHorizontalFlip(p=0.5),
            VideoNormalize()
        ])
    else:
        return ComposeVideoTransforms([
            VideoResize(frame_size),
            VideoNormalize()
        ])
