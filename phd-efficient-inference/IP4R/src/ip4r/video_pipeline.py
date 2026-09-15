import cv2
import numpy as np
from ultralytics import YOLO

import sys
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# Add repo root to sys.path so we can import server_v2 modules
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))
from server_v2.lcd_crop import detect_lcd

# Import Inspector (21 ROIs) Stage 2
from ip4r.pipeline import Inspector
from ip4r.config import Config
from ip4r.preprocess import preprocess
from ip4r.registration import register

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _predict(model, device, crop_bgr: np.ndarray) -> float:
    img  = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x    = _TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze().item()
    return float(torch.sigmoid(torch.tensor(logit)).item())

def _map_roi_quad(bbox: tuple, H_inv: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    mapped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_inv)
    return mapped.reshape(-1, 2).astype(int)

class VideoInspector:
    def __init__(self, yolo_model_path: str, old_inspector=None, fps: int = 12):
        self.fps = fps
        self.stage = 1
        self.defect_detected = False
        self.failure_reason = ""
        self.debug_saved = False
        self.saved_seconds = set()
        
        self.debug_dir = Path("debug_failures")
        self.debug_dir.mkdir(exist_ok=True)
        
        # Load the 21-ROI Inspector
        self.cfg = Config("config.yaml")
        self.inspector = Inspector(self.cfg)
        
        # Load YOLO
        self.yolo = YOLO(yolo_model_path)
        
        # Load EfficientNet-B0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = torch.nn.Linear(self.efficientnet.classifier[1].in_features, 1)
        model_path = _repo_root / "models" / "dts_p2v2_best.pth"
        ckpt = torch.load(str(model_path), map_location=self.device)
        self.efficientnet.load_state_dict(ckpt["state_dict"])
        self.efficientnet.eval().to(self.device)
        
        # We expect exactly 5 unique macro blocks on the screen
        self.EXPECTED_CLASSES = {
            "top_left_block",
            "top_right_block",
            "middle_block",
            "signal_icon",
            "footer_digits"
        }
        
        # Checklist State Tracking
        self.checklist = {
            label: {'detected': False, 'best_conf': 0.0, 'box': None}
            for label in self.EXPECTED_CLASSES
        }
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[bool, str, np.ndarray]:
        """
        Processes a single frame through the state machine.
        Returns: (is_valid, status_message, annotated_frame)
        """
        # Create copies for drawing and masking
        annotated_frame = frame.copy()
        masked_frame = frame.copy()
        
        if self.stage == 1:
            # ==========================================
            # WAIT FOR 3 SECONDS BEFORE STARTING YOLO
            # ==========================================
            if frame_idx < self.fps * 3:
                status_msg = f"Waiting... ({frame_idx/self.fps:.1f}s / 3.0s)"
                cv2.putText(annotated_frame, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return True, status_msg, annotated_frame

            # ==========================================
            # STAGE 1: YOLO STARTUP SEQUENCE VALIDATION
            # ==========================================
            results = self.yolo(frame, verbose=False)[0]
            
            low_accuracy_detected = False
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names[cls_id]
                conf = float(box.conf[0])
                
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls_name in self.checklist:
                    if conf >= 0.80:
                        self.checklist[cls_name]['detected'] = True
                    else:
                        low_accuracy_detected = True
                        
                    if conf > self.checklist[cls_name]['best_conf']:
                        self.checklist[cls_name]['best_conf'] = conf
                        self.checklist[cls_name]['box'] = (x1, y1, x2, y2)
                
                # Draw boxes for visual feedback (Green if >= 0.80, Red if < 0.80)
                color = (0, 255, 0) if conf >= 0.80 else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # Draw Global Checklist Summary on the right side
            summary_x = max(10, annotated_frame.shape[1] - 250)
            summary_y = 60
            cv2.putText(annotated_frame, "Global Checklist:", (summary_x, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            summary_y += 25
            for cls_name, state in self.checklist.items():
                checked = "[x]" if state['detected'] else "[ ]"
                conf = state['best_conf']
                color = (0, 255, 0) if state['detected'] else (200, 200, 200)
                cv2.putText(annotated_frame, f"{checked} {cls_name}: {conf:.2f}", (summary_x, summary_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                summary_y += 20
                
            # ==========================================
            # 15 SECOND FORCED TRANSITION
            # ==========================================
            if frame_idx >= self.fps * 15:
                # Save YOLO detections screenshot
                cv2.imwrite(str(self.debug_dir / "01_yolo_detections_with_accuracy.jpg"), annotated_frame)
                if low_accuracy_detected:
                    cv2.imwrite(str(self.debug_dir / "01_low_accuracy_detected.jpg"), annotated_frame)
                    
                # 2. Masking Stage (Anomaly Detection)
                # Draw WHITE rectangles over all accumulated boxes from the checklist (even if partial)
                for state in self.checklist.values():
                    if state['box']:
                        x1, y1, x2, y2 = state['box']
                        padding = 5
                        cv2.rectangle(masked_frame, (max(0, x1 - padding), max(0, y1 - padding)), 
                                                    (min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)), 
                                                    (255, 255, 255), -1)
                                                    
                # Save the white-masked screen
                cv2.imwrite(str(self.debug_dir / "02_white_masked_screen.jpg"), masked_frame)
                
                # Force Transition to Stage 2 (Process with EfficientNet logic)
                self.stage = 2
                self.debug_saved = False # Reset for stage 2 screenshots
                
                # Check if it was actually complete
                all_checked_off = all(state['detected'] for state in self.checklist.values())
                if not all_checked_off:
                    msg = "15s Reached: Checklist INCOMPLETE. Forcing EfficientNet..."
                else:
                    msg = "15s Reached: Checklist COMPLETE. Transitioning to EfficientNet..."
                    
                return True, msg, annotated_frame
                    
            status_msg = f"Stage 1 (YOLO): Accumulating... ({sum(1 for s in self.checklist.values() if s['detected'])}/5)"
            cv2.putText(annotated_frame, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return True, status_msg, annotated_frame
            
        elif self.stage == 2:
            # ==========================================
            # STAGE 2: EFFICIENT-NET VALIDATION + 21-ROI INSPECTION
            # ==========================================
            # 1. Apply masks to current frame and draw green boxes on display frame
            masked_frame = frame.copy()
            for cls_name, state in self.checklist.items():
                if state['box']:
                    x1, y1, x2, y2 = state['box']
                    padding = 5
                    # Mask with white for the CNN
                    cv2.rectangle(masked_frame, (max(0, x1 - padding), max(0, y1 - padding)), 
                                                (min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)), 
                                                (255, 255, 255), -1)
            
            # 2. Crop to LCD for EfficientNet
            crop, _ = detect_lcd(masked_frame)
            if crop is None:
                msg = "Phase 2: LCD not detected for crop"
                cv2.putText(annotated_frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return False, msg, annotated_frame
                
            # 3. Predict EfficientNet probability
            prob = _predict(self.efficientnet, self.device, crop)
            
            # 4. Draw Probability on main frame
            verdict = "PASS" if prob >= 0.5 else "FAIL"
            color = (0, 255, 0) if verdict == "PASS" else (0, 0, 255)
            msg = f"Phase 2 (EfficientNet): {verdict} (Prob: {prob*100:.1f}%)"
            
            cv2.putText(annotated_frame, msg, (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 5. Run the 21-ROI Inspector (Legacy CV Check)
            insp_result = self.inspector.inspect_array(frame, "video_frame")
            
            # Calculate homography to project 21 ROI boxes onto the screen
            proc = preprocess(frame, self.cfg)
            bbox = self.inspector._get_remote_bbox(frame)
            if bbox:
                x, y, w, h = bbox
                proc_crop = proc[y:y+h, x:x+w]
                _, reg = register(proc_crop, self.inspector.golden_proc, self.cfg)
                H_crop = reg.get("homography")
                if H_crop is not None:
                    T_inv = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
                    H_trig = H_crop @ T_inv
                else:
                    H_trig = None
            else:
                _, reg = register(proc, self.inspector.golden_proc, self.cfg)
                H_trig = reg.get("homography")
            
            # Draw the 21 bounding boxes
            if H_trig is not None:
                H_inv = np.linalg.inv(H_trig)
                for r in insp_result.roi_results:
                    roi_color = (0, 255, 0) if r.passed else (0, 0, 255)
                    quad = _map_roi_quad(r.bbox, H_inv)
                    cv2.polylines(annotated_frame, [quad], isClosed=True, color=roi_color, thickness=2 if r.passed else 3)
                    top_pt = tuple(quad[np.argmin(quad[:, 1])])
                    label = r.name.replace("_", " ")
                    cv2.putText(annotated_frame, label, (top_pt[0], max(20, top_pt[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, roi_color, 1, cv2.LINE_AA)
            
            # Draw the raw white-masked crop as a picture-in-picture
            crop_resized = cv2.resize(crop, (240, 320))
            annotated_frame[10:330, 10:250] = crop_resized
            cv2.rectangle(annotated_frame, (10, 10), (250, 330), color, 2)
            
            # Save screenshots at 15s, 16s, 17s, 18s
            current_sec = frame_idx // self.fps
            if current_sec in [15, 16, 17, 18] and current_sec not in self.saved_seconds:
                cv2.imwrite(str(self.debug_dir / f"03_stage2_cnn_frame_{current_sec}s.jpg"), annotated_frame)
                cv2.imwrite(str(self.debug_dir / f"04_stage2_cnn_crop_{current_sec}s.jpg"), crop)
                self.saved_seconds.add(current_sec)
                
            return (prob >= 0.5), msg, annotated_frame
