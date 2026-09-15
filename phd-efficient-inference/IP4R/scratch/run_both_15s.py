import cv2
import os
import shutil
from pathlib import Path
from ip4r.config import Config, REPO_ROOT
from ip4r.pipeline import Inspector
from ip4r.video_pipeline import VideoInspector

def process_video(video_path, output_dir, inspector_cfg):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    yolo_model = str(REPO_ROOT / "data" / "macro_dataset" / "runs" / "macro_test" / "weights" / "best.pt")
    
    inspector = VideoInspector(yolo_model_path=yolo_model, fps=12)
    # Set unique debug dir so screenshots don't collide
    inspector.debug_dir = output_dir / f"debug_{video_path.stem}"
    inspector.debug_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out_video_path = output_dir / f"{video_path.stem}_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    
    print(f"Processing {video_path.name}...")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        is_valid, msg, annotated = inspector.process_frame(frame, frame_idx)
        out.write(annotated)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"Done! Saved video to {out_video_path}")

def main():
    cfg = Config.load(REPO_ROOT / "config" / "default.yaml")
    
    good_video = r"D:\New Project\Aug-28-2026\12 FPS\GOOD\video@12FPS_20260828_011156.mp4"
    bad_video = r"D:\New Project\Aug-28-2026\12 FPS\NOT GOOD\video@12FPS_20260828_014317.mp4"
    
    out_dir = r"D:\New Project\Forced_15s_Results"
    
    process_video(good_video, out_dir, cfg)
    process_video(bad_video, out_dir, cfg)

if __name__ == "__main__":
    main()
