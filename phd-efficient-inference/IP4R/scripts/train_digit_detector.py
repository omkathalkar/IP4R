"""
Phase 1 — Train Stage 1 Digit Detector
======================================
Trains a simple binary CNN (ON vs OFF) for the 5 stage-1 ROIs.

Input:
  data/training/*/on/*.png  -> Class 1 (ON)
  data/training/*/off/*.png -> Class 0 (OFF)

Output:
  models/stage1_detector.pt
"""

import os
import glob
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 1. Dataset Loader
# ---------------------------------------------------------
class DigitPresenceDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        
        # We only care about the 5 target ROIs for Stage 1
        target_rois = [
            "clock_digits", 
            "fan_speed_bars", 
            "foot_display_digits", 
            "temperature_digits", 
            "timer_off_digits"
        ]

        root_path = Path("D:/New Project/IP4R/phd-efficient-inference/IP4R/data/training")
        
        # Gather 'on' (1) and 'off' (0) images
        for roi in target_rois:
            roi_path = root_path / roi
            if not roi_path.exists():
                continue
                
            on_images = glob.glob(str(roi_path / "on" / "*.jpg")) + glob.glob(str(roi_path / "on" / "*.png"))
            off_images = glob.glob(str(roi_path / "off" / "*.jpg")) + glob.glob(str(roi_path / "off" / "*.png"))
            
            for img_path in on_images:
                self.samples.append((img_path, 1))
            for img_path in off_images:
                self.samples.append((img_path, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        
        # Binarize with Adaptive Thresholding to handle uneven LCD lighting
        img_arr = np.array(image)
        binarized = cv2.adaptiveThreshold(
            img_arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
        )
        image = Image.fromarray(binarized)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ---------------------------------------------------------
# 2. Simple CNN Architecture
# ---------------------------------------------------------
class SimpleDigitCNN(nn.Module):
    def __init__(self):
        super(SimpleDigitCNN, self).__init__()
        # Input: 1 x 64 x 64
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def main():
    model_out = Path("D:/New Project/IP4R/phd-efficient-inference/IP4R/models/stage1_detector.pt")
    
    # Data Augmentation to prevent overfitting and handle bbox drift
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=2, translate=(0.1, 0.1)), # CRITICAL for translation invariance!
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset
    dataset = DigitPresenceDataset(transform=transform_train)
    if len(dataset) == 0:
        print("ERROR: No images found in the 'on' or 'off' folders.")
        return
        
    print(f"Loaded {len(dataset)} images for training.")
    
    # Check class balance
    on_count = sum(1 for _, label in dataset.samples if label == 1)
    off_count = sum(1 for _, label in dataset.samples if label == 0)
    print(f"Class Balance -> ON: {on_count} | OFF: {off_count}")
    
    # Handle severe imbalance with weighted loss (since we have ~800 ON and ~130 OFF)
    weight_for_0 = (on_count + off_count) / (2.0 * off_count) if off_count > 0 else 1.0
    weight_for_1 = (on_count + off_count) / (2.0 * on_count) if on_count > 0 else 1.0
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)
    print(f"Class weights: [OFF: {weight_for_0:.2f}, ON: {weight_for_1:.2f}]")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Setup model, loss, optimizer
    model = SimpleDigitCNN()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    epochs = 10
    print("\nStarting Training...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(dataloader):.4f} - Accuracy: {epoch_acc:.2f}%")

    # Save
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(model_out))
    print(f"\nSUCCESS: Model saved to {model_out}")

if __name__ == "__main__":
    main()
