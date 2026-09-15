import argparse
import sys
import cv2
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

ROOT = Path("D:/New Project/IP4R/phd-efficient-inference/IP4R")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ip4r.config import Config
from ip4r.pipeline import Inspector
from train_digit_detector import SimpleDigitCNN

def get_proportional_rois(frame, inspector, target_names):
    """Map specific ROIs from golden to frame using proportional bbox scaling."""
    bbox = inspector._get_remote_bbox(frame)
    if not bbox: return None
    vx, vy, vw_box, vh_box = bbox
    
    gh, gw = inspector.golden_proc.shape[:2]
    gx, gy, gw_box, gh_box = 124, 5, 368, 475
    
    mapped_rois = {}
    for roi in inspector.rois:
        if roi.name not in target_names: continue
        
        abs_x = roi.x * gw
        abs_y = roi.y * gh
        abs_w = roi.w * gw
        abs_h = roi.h * gh
        
        frac_x = (abs_x - gx) / gw_box
        frac_y = (abs_y - gy) / gh_box
        frac_w = abs_w / gw_box
        frac_h = abs_h / gh_box
        
        final_x = int(vx + frac_x * vw_box)
        final_y = int(vy + frac_y * vh_box)
        final_w = int(frac_w * vw_box)
        final_h = int(frac_h * vh_box)
        
        mapped_rois[roi.name] = (final_x, final_y, final_w, final_h)
        
    return mapped_rois

def process_image(image_path, out_dir, cfg, inspector, model, device, transform, target_names):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Could not read {image_path.name}")
        return
        
    mapped_rois = get_proportional_rois(frame, inspector, target_names)
    if not mapped_rois:
        print(f"[WARNING] Could not find remote in {image_path.name}")
        return
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_frame = frame.copy()
    
    for name, (x, y, w, h) in mapped_rois.items():
        crop = gray_frame[y:y+h, x:x+w]
        if crop.size == 0: continue
            
        binarized = cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
        )
            
        pil_img = Image.fromarray(binarized)
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor_img)
            probs = torch.softmax(out, dim=1)
            p_on = probs[0][1].item()
            
        # Draw red if OFF (<50%), Green if ON (>50%)
        if p_on > 0.50:
            color = (0, 255, 0) # Green
        else:
            color = (0, 0, 255) # Red
            
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(output_frame, f"{name}: {p_on*100:.0f}%", (x, max(10, y-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
    out_path = out_dir / image_path.name
    cv2.imwrite(str(out_path), output_frame)
    print(f"Processed {image_path.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder_path", help="Path to input folder containing images")
    args = ap.parse_args()

    in_dir = Path(args.folder_path)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] Invalid directory: {in_dir}")
        return
        
    out_dir = in_dir / "output"
    out_dir.mkdir(exist_ok=True)
    
    # Setup
    cfg = Config.load(str(ROOT / "config" / "default.yaml"))
    inspector = Inspector(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ROOT / "models" / "stage1_detector.pt"
    model = SimpleDigitCNN().to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    target_names = ["stage1_timer_off", "stage1_timer_on", "stage1_temperature_tens", "stage1_fan_speed_bars", "stage1_foot_display"]
    
    valid_exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    
    print(f"Found {len(images)} images to process in {in_dir}")
    for img_path in images:
        process_image(img_path, out_dir, cfg, inspector, model, device, transform, target_names)
        
if __name__ == "__main__":
    main()
