# This file defines the architecture of your custom model.

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# 1. The Configuration Class
class MnistCNNConfig(PretrainedConfig):
    model_type = "kenil_mnist_cnn"

    def __init__(self, image_size=28, num_channels=1, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes

# 2. The Model Class
class MnistCNN(PreTrainedModel):
    config_class = MnistCNNConfig
    def __init__(self, config):
        super().__init__(config)
        self.network = nn.Sequential(
            nn.Conv2d(config.num_channels, 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(3136, config.num_classes)

    def forward(self, pixel_values):
        x = self.network(pixel_values)
        x = self.flatten(x)
        logits = self.lin(x)
        return {"logits": logits}