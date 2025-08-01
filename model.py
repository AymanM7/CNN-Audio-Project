import torch
import torch.nn as nn


try:
    # For torchvision >= 0.13
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
except ImportError:
    # For older torchvision
    from torchvision.models import resnet18
    weights = "IMAGENET1K_V1"


class AudioCNN(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # Load pretrained ResNet18
        self.resnet = resnet18(weights=weights)
        # Change first conv layer to accept 1 channel (spectrograms)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)


    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        # ResNet expects at least 224x224, so resize if needed
        if x.shape[2] < 224 or x.shape[3] < 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.resnet(x)  

