"""
Phase 2 — Prototype Video Scanner (Stage 1 Vertical Slice)
==========================================================
Verifies that we can correctly detect the build-up sequence of the 5
Stage 1 elements over the first 15 seconds of a video.

Usage:
  python scripts/test_stage1_sequence.py "path/to/video.mp4"
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# Load project internals
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ip4r.config import Config
from ip4r.pipeline import Inspector
from ip4r.preprocess import preprocess
from ip4r.registration import register

# ---------------------------------------------------------
# Model Definition (Must match train_digit_detector.py)
# ---------------------------------------------------------
class SimpleDigitCNN(nn.Module):
    def __init__(self):
        super(SimpleDigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------
class Stage1Scanner:
    def __init__(self, model_path: str, inspector: Inspector, cfg: Config):
        self.cfg = cfg
        self.inspector = inspector
        
        # Load Model
        self.model = SimpleDigitCNN()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # The 5 DIGIT/ICON elements we want to monitor for Stage 1.
        # Note: We use 'timer_on' for the top-right digits (which you called clock_digits)
        # and 'temperature_tens' for the middle-left digits.
        self.target_names = [
            "timer_off",       # Top-left digits
            "timer_on",        # Top-right digits (clock_digits)
            "temperature_tens",# Middle-left digits (temperature_digits)
            "fan_speed_bars",  # Middle-right bars
            "foot_display"     # Bottom digits
        ]
        
        # Extract the specific ROI objects from inspector
        roi_map = {r.name: r for r in inspector.rois}
        self.rois = [roi_map[n] for n in self.target_names if n in roi_map]
        
        # Timeline tracking: {roi_name: time_first_seen_in_seconds}
        self.timeline = {r.name: None for r in self.rois}
        
    def get_proportional_rois(self, frame):
        """Map ROIs from golden to frame using proportional bbox scaling."""
        bbox = self.inspector._get_remote_bbox(frame)
        if not bbox:
            return None
            
        vx, vy, vw_box, vh_box = bbox
        
        # Golden reference constants (derived from lcd_all_on.jpg)
        gh, gw = self.inspector.golden_proc.shape[:2]  # 480, 640
        gx, gy, gw_box, gh_box = 124, 5, 368, 475      # Golden remote bbox
        
        mapped_rois = {}
        for roi in self.rois:
            # Absolute golden coordinates
            abs_x = roi.x * gw
            abs_y = roi.y * gh
            abs_w = roi.w * gw
            abs_h = roi.h * gh
            
            # Fraction of golden remote size
            frac_x = (abs_x - gx) / gw_box
            frac_y = (abs_y - gy) / gh_box
            frac_w = abs_w / gw_box
            frac_h = abs_h / gh_box
            
            # Absolute frame coordinates
            final_x = int(vx + frac_x * vw_box)
            final_y = int(vy + frac_y * vh_box)
            final_w = int(frac_w * vw_box)
            final_h = int(frac_h * vh_box)
            
            mapped_rois[roi.name] = (final_x, final_y, final_w, final_h)
            
        return mapped_rois
        
    def crop_and_predict(self, frame, mapped_rois):
        """Crop the mapped ROIs from the frame and predict ON/OFF."""
        on_this_frame = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for name, (x, y, w, h) in mapped_rois.items():
            crop = gray_frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
                
            # Binarize with Adaptive Thresholding to handle uneven LCD lighting
            binarized = cv2.adaptiveThreshold(
                crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
            )
                
            # Convert to PIL 'L' mode to match training data
            pil_img = Image.fromarray(binarized)
            tensor_img = self.transform(pil_img).unsqueeze(0)
            
            with torch.no_grad():
                out = self.model(tensor_img)
                probs = torch.softmax(out, dim=1)
                
            p_on = probs[0][1].item()
            if p_on > 0.1: # Print anything that has even a slight chance
                pass # print(f"DEBUG: {name} prob_ON={p_on:.3f}")
                
            # Confidence gate
            if p_on > 0.50:  # Lowered threshold to see if it detects anything
                on_this_frame.append(name)
                
        return on_this_frame

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()
    
    cfg = Config.load(str(ROOT / "config" / "default.yaml"))
    inspector = Inspector(cfg)
    model_path = str(ROOT / "models" / "stage1_detector.pt")
    
    scanner = Stage1Scanner(model_path, inspector, cfg)
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n--- Scanning Video for Stage 1 Elements ---")
    print(f"Video: {args.video} ({total_frames} frames @ {fps:.1f} fps)")
    print(f"Tracking: {scanner.target_names}\n")
    
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        t_sec = fi / fps
        
        # Start detection strictly after 3 seconds (as requested)
        if t_sec < 3.0:
            fi += 1
            continue
            
        # We only need to scan every 2-3 frames to be efficient
        if fi % 3 == 0:
            mapped_rois = scanner.get_proportional_rois(frame)
            if mapped_rois is not None:
                on_rois = scanner.crop_and_predict(frame, mapped_rois)
                
                # Update timeline and save individual snapshots
                for r in on_rois:
                    if scanner.timeline[r] is None:
                        scanner.timeline[r] = t_sec
                        print(f"[{t_sec:05.2f}s] DETECTED: {r}")
                        
                        # Save a snapshot for THIS specific detection
                        result_frame = frame.copy()
                        x, y, w, h = mapped_rois[r]
                        cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        
                        out_path = str(ROOT / "scratch" / f"{r}_detected.jpg")
                        cv2.imwrite(out_path, result_frame)
                        print(f"  -> Saved snapshot to: {out_path}")
                        
                # If all are found, we can stop for this vertical slice test
                if all(v is not None for v in scanner.timeline.values()):
                    print(f"\nSUCCESS: All 5 Stage 1 elements detected!")
                    break
        
        fi += 1
        
    cap.release()
    
    print("\n--- FINAL TIMELINE ---")
    for r, t in scanner.timeline.items():
        if t is None:
            print(f"FAIL: {r:18s}: NEVER DETECTED")
        else:
            print(f"PASS: {r:18s}: appeared at {t:.2f}s")

if __name__ == "__main__":
    main()
