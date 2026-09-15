import cv2
import os
from pathlib import Path
from ip4r.config import Config, REPO_ROOT
from ip4r.pipeline import Inspector
from ip4r.video_pipeline import VideoInspector

def main():
    videos_dir = Path(r"D:\New Project\Aug-28-2026\12 FPS\NOT GOOD")
    output_dir = Path(r"D:\New Project\Batch_Results")
    output_dir.mkdir(exist_ok=True)
    
    cfg = Config.load(REPO_ROOT / "config" / "default.yaml")
    try:
        old_inspector = Inspector(cfg)
    except Exception as e:
        print(f"Warning: Could not load old Inspector: {e}")
        old_inspector = None

    yolo_model = str(REPO_ROOT / "data" / "macro_dataset" / "runs" / "macro_test" / "weights" / "best.pt")
    
    video_files = list(videos_dir.glob("*.mp4"))
    limit = 3 # Processing the first 3 videos
    
    for i, video_path in enumerate(video_files[:limit]):
        print(f"Processing video {i+1}/{limit}: {video_path.name}")
        
        inspector = VideoInspector(yolo_model_path=yolo_model, old_inspector=old_inspector, fps=12)
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Setup Video Writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out_video_path = output_dir / f"result_{video_path.name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            is_valid, status_msg, annotated = inspector.process_frame(frame, frame_idx)
            
            # Write the annotated frame to the output video
            out.write(annotated)
            
            if not is_valid:
                print(f"  -> Failed at frame {frame_idx}: {status_msg}")
                
            frame_idx += 1
            
        cap.release()
        out.release()
        print(f"  -> Video saved to {out_video_path}")
        
    print(f"\nDone! Videos saved in: {output_dir}")

if __name__ == "__main__":
    main()
