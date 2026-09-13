"""EfficientNet-B0 binary PASS/FAIL classifier for 480×640 LCD splash crops.

Label convention (used throughout v6a):
  0 = GOOD  (PASS)
  1 = NOT_GOOD  (FAIL)

Model output: single logit → sigmoid → prob_fail ∈ [0, 1]
Verdict: FAIL if prob_fail > threshold (default 0.5)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

INPUT_SIZE = 224

# Normalisation matching ImageNet pre-training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# Training augmentation: keep the LCD in view (small crops + rotation),
# vary lighting (colorjitter), simulate jitter (affine translate).
# No horizontal flip — the LCD layout is not horizontally symmetric.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 24, INPUT_SIZE + 24)),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
    return net


def freeze_backbone(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: nn.Module, path: str | Path, meta: dict | None = None) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "meta":       meta or {},
    }, str(path))


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = get_device()
    net = build_model(pretrained=False)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval().to(device)
    return net


def predict_crop(
    crop_bgr: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> float:
    """Return prob_fail ∈ [0, 1] for a 480×640 BGR LCD crop."""
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil     = Image.fromarray(img_rgb)
    tensor  = INFER_TRANSFORM(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor).squeeze()
    return float(torch.sigmoid(logit).item())
