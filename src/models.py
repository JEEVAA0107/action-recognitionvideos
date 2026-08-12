import torch
import torch.nn as nn
import torchvision.models.video as video_models
from config import Config

class ActionRecognition3DCNN(nn.Module):
    """
    Spatiotemporal 3D CNN Action Recognition Model.
    Supports torchvision video backbone models:
    - R(2+1)D 18-layer (r2plus1d_18)
    - ResNet 3D 18-layer (r3d_18)
    - Mixed 3D 18-layer (mc3_18)
    """
    def __init__(self, num_classes=5, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(ActionRecognition3DCNN, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        print(f"[INFO] Initializing {model_name} (Pretrained={pretrained}) for {num_classes} action classes...")

        if model_name == "r2plus1d_18":
            weights = video_models.R2Plus1D_18_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.r2plus1d_18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif model_name == "r3d_18":
            weights = video_models.R3D_18_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.r3d_18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        elif model_name == "mc3_18":
            weights = video_models.MC3_18_Weights.DEFAULT if pretrained else None
            self.backbone = video_models.mc3_18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported model architecture: {model_name}. Choose from ('r2plus1d_18', 'r3d_18', 'mc3_18')")

    def forward(self, x):
        """
        Forward pass.
        Expected input shape x: (Batch_Size, Channels, Frames, Height, Width) -> (B, 3, 16, 112, 112)
        Output shape: (Batch_Size, Num_Classes)
        """
        return self.backbone(x)


def get_model(num_classes=5, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
    """
    Factory function to construct and return the action recognition model.
    """
    model = ActionRecognition3DCNN(num_classes=num_classes, model_name=model_name, pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Model Architecture Smoke Test
    print("[TEST] Testing ActionRecognition3DCNN Forward Pass...")
    device = "cpu"
    dummy_model = get_model(num_classes=5, pretrained=False).to(device)
    dummy_input = torch.randn(2, 3, 16, 112, 112).to(device) # (B=2, C=3, T=16, H=112, W=112)
    dummy_output = dummy_model(dummy_input)
    print(f"[TEST] Input shape: {dummy_input.shape}")
    print(f"[TEST] Output shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 5), "Output shape mismatch!"
    print("[TEST] Model test passed successfully!")
