import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------
# CNN Model Definition (Must match train_digit_detector.py)
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
# Main Pipeline
# ---------------------------------------------------------
def main():
    root_dir = Path(r"D:\New Project\IP4R\phd-efficient-inference\IP4R")
    video_path = r"D:\New Project\Aug-28-2026\12 FPS\NOT GOOD\video@12FPS_20260828_014049.mp4"
    
    # 1. Load YOLO Model (Stage 1)
    yolo_model_path = root_dir / "data" / "macro_dataset" / "runs" / "macro_test" / "weights" / "best.pt"
    print(f"Loading YOLO model from: {yolo_model_path}")
    yolo_model = YOLO(str(yolo_model_path))
    
    # 2. Load CNN Model (Stage 2)
    cnn_model_path = root_dir / "models" / "stage1_detector.pt"
    print(f"Loading CNN model from: {cnn_model_path}")
    cnn_model = SimpleDigitCNN()
    cnn_model.load_state_dict(torch.load(str(cnn_model_path), map_location="cpu"))
    cnn_model.eval()
    
    # CNN Transforms
    cnn_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 3. Setup output folder
    out_dir = root_dir / "runs_inference" / "footer_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Initialize Checklist
    checklist = {
        "footer_digits": False,
        "middle_block": False,
        "signal_icon": False,
        "top_left_block": False,
        "top_right_block": False
    }
    
    # Track if we have already extracted the footer images
    extracted_footer = False
    
    cap = cv2.VideoCapture(video_path)
    print("\nStarting Video Scan...")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t_sec = frame_idx / fps
        
        # 1. Skip the first 3 seconds
        if t_sec < 3.0:
            frame_idx += 1
            continue
            
        # Run YOLO detection
        results = yolo_model(frame, verbose=False)[0]
        
        # Track max confidence for each class in this frame (for text drawing if < 0.8)
        frame_confs = {}
        
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            
            if cls_name not in frame_confs or conf > frame_confs[cls_name]['conf']:
                frame_confs[cls_name] = {'conf': conf, 'box': box}
                
            # Check off the item in our checklist if >= 80%
            if conf >= 0.80:
                if cls_name in checklist and not checklist[cls_name]:
                    checklist[cls_name] = True
                    print(f"[{t_sec:.1f}s] Checked off: {cls_name} (conf: {conf:.2f})")
                
                # If footer_digits is detected for the first time, process it!
                if cls_name == "footer_digits" and not extracted_footer:
                    print(f"--- FOOTER DIGITS DETECTED at {t_sec:.1f}s! Triggering Masking & CNN... ---")
                    extracted_footer = True
                    
                    # Create annotated screenshot
                    annotated_frame = frame.copy()
                    
                    # Draw all detections in this frame according to rules
                    for name, data in frame_confs.items():
                        c = data['conf']
                        b = data['box']
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        label = f"{name} {c*100:.1f}%"
                        
                        if c >= 0.80:
                            # Draw box + label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            # Draw label only (no box)
                            cv2.putText(annotated_frame, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # 1. Save Full Screenshot (Annotated)
                    cv2.imwrite(str(out_dir / "1_full_screenshot.jpg"), annotated_frame)
                    
                    # 2. Mask/Crop the footer block (from original frame, not annotated)
                    fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                    crop = frame[fy1:fy2, fx1:fx2]
                    cv2.imwrite(str(out_dir / "2_masked_crop.jpg"), crop)
                    
                    # 3. Prepare for CNN (Grayscale + Adaptive Threshold)
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    binarized = cv2.adaptiveThreshold(
                        gray_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
                    )
                    cv2.imwrite(str(out_dir / "3_cnn_input.jpg"), binarized)
                    
                    # Run CNN Inference
                    pil_img = Image.fromarray(binarized)
                    tensor_img = cnn_transform(pil_img).unsqueeze(0)
                    
                    with torch.no_grad():
                        out = cnn_model(tensor_img)
                        probs = torch.softmax(out, dim=1)
                        
                    p_off = probs[0][0].item()
                    p_on = probs[0][1].item()
                    
                    status = "ON" if p_on > 0.50 else "OFF"
                    
                    # Save results to text
                    result_text = f"Footer Digits Inspection Result\n" \
                                  f"===============================\n" \
                                  f"Status: {status}\n" \
                                  f"Probability ON: {p_on:.4f}\n" \
                                  f"Probability OFF: {p_off:.4f}\n" \
                                  f"YOLO Confidence: {conf:.4f}\n"
                    
                    with open(str(out_dir / "cnn_result.txt"), "w") as f:
                        f.write(result_text)
                        
                    print(f"CNN Result: {status} (Prob ON: {p_on:.2f})")
                    print(f"Images and result saved to {out_dir}")
        
        # Check if phase 1 checklist is completely done
        if all(checklist.values()):
            print("\nSUCCESS: Phase 1 Detection Completed! All 5 labels were checked off.")
            break
            
        frame_idx += 1
        
    cap.release()
    
    if not all(checklist.values()):
        print("\nFAIL: Video ended without detecting all 5 labels.")
        missing = [k for k, v in checklist.items() if not v]
        print(f"Missing labels: {missing}")

if __name__ == "__main__":
    main()
