import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from transforms import get_video_transforms
from models import get_model

def predict_video_action(video_path, checkpoint_path=Config.BEST_MODEL_PATH, output_path=None, device_str=Config.DEVICE):
    """
    Runs action recognition inference on an input video file and saves annotated output.
    """
    device = torch.device(device_str)

    if not os.path.exists(video_path):
        print(f"[ERROR] Input video not found at: {video_path}")
        return

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Model checkpoint not found at: {checkpoint_path}")
        return

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint.get('classes', Config.CLASSES)
    model_name = checkpoint.get('model_name', Config.MODEL_NAME)

    # Initialize Model
    model = get_model(num_classes=len(classes), model_name=model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Read Video Frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    raw_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame.copy())
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        print(f"[ERROR] Could not extract frames from {video_path}")
        return

    # Uniformly sample 16 frames
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, Config.NUM_FRAMES, dtype=int)
    sampled_frames = [frames[i] for i in indices]

    # Preprocess Tensor
    transform = get_video_transforms(is_train=False, frame_size=(Config.FRAME_HEIGHT, Config.FRAME_WIDTH))
    frames_uint8 = torch.from_numpy(np.array(sampled_frames, dtype=np.uint8))
    clip_tensor = transform(frames_uint8).unsqueeze(0).to(device) # Shape: (1, C, T, H, W)

    # Forward Pass
    with torch.no_grad():
        logits = model(clip_tensor)
        probabilities = F.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)

    pred_class = classes[pred_idx.item()]
    conf_pct = confidence.item() * 100.0

    print("\n" + "="*50)
    print(f"       INFERENCE RESULT: {os.path.basename(video_path)}")
    print("="*50)
    print(f" Predicted Action : {pred_class.upper()}")
    print(f" Confidence Score : {conf_pct:.2f}%")
    print("="*50 + "\n")

    # Annotate Video Frames
    if output_path is None:
        output_path = os.path.join(Config.OUTPUT_DIR, f"annotated_{os.path.basename(video_path)}")

    height, width, _ = raw_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 15, (width, height))

    label_text = f"Action: {pred_class.upper()} ({conf_pct:.1f}%)"

    for frame in raw_frames:
        # Draw background rectangle for readability
        cv2.rectangle(frame, (10, 10), (width - 10, 50), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)

    out.release()
    print(f"[INFO] Saved annotated prediction video to: {output_path}")

    # Automatically open video on Windows if supported
    try:
        if os.name == 'nt' and os.path.exists(output_path):
            os.startfile(output_path)
            print(f"[INFO] Opening annotated video player...")
    except Exception as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Action Recognition Inference on a Video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint_path", type=str, default=Config.BEST_MODEL_PATH, help="Path to saved model checkpoint")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="Device (cuda or cpu)")

    args = parser.parse_args()
    predict_video_action(args.video_path, args.checkpoint_path, args.output_path, args.device)
