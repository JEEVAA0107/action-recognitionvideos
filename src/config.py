import os
import torch

class Config:
    # Random Seed for Reproducibility
    SEED = 42

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Model Weights Path
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Video Preprocessing & Sampling Settings
    NUM_FRAMES = 16          # Temporal frames per clip
    FRAME_HEIGHT = 112       # Spatial height
    FRAME_WIDTH = 112        # Spatial width
    FPS = 15                 # Sampling frame rate

    # Model Parameters
    MODEL_NAME = "r2plus1d_18"  # Options: 'r3d_18', 'r2plus1d_18', 'mc3_18'
    PRETRAINED = True

    # Training Hyperparameters
    BATCH_SIZE = 8
    NUM_WORKERS = 0          # Set to 0 for Windows stability
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Action Class Labels for the UCF101 subset
    CLASSES = [
        "Archery", "Basketball", "Biking", "Bowling", "BoxingPunching", 
        "Diving", "Fencing", "GolfSwing", "HorseRiding", "SoccerJuggling"
    ]

    @classmethod
    def setup_directories(cls):
        """Ensure necessary output & checkpoint directories exist."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
