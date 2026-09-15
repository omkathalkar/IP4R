import cv2
import argparse
from pathlib import Path
from ip4r.config import Config, REPO_ROOT
from ip4r.pipeline import Inspector
from ip4r.video_pipeline import VideoInspector

def main():
    parser = argparse.ArgumentParser(description="Run the Two-Stage YOLO + Rules Engine Pipeline on a Video")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--yolo-model", type=str, 
                        default=str(REPO_ROOT / "data" / "macro_dataset" / "runs" / "macro_test" / "weights" / "best.pt"),
                        help="Path to the trained YOLO best.pt model")
    parser.add_argument("--out", type=str, default="inspection_result.mp4", help="Path to save the annotated video")
    args = parser.parse_args()
    
    print(f"Loading Configuration from {REPO_ROOT / 'config' / 'default.yaml'}...")
    cfg = Config.load(REPO_ROOT / "config" / "default.yaml")
    
    print(f"Initializing Stage 1 (YOLO) and Stage 2 (EfficientNet) with model: {args.yolo_model}...")
    video_inspector = VideoInspector(yolo_model_path=args.yolo_model, fps=12)
    
    print(f"\nOpening Video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
    
    frame_idx = 0
    final_result = True
    final_reason = "PASS"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        is_valid, status_msg, annotated_frame = video_inspector.process_frame(frame, frame_idx)
        
        # Write to output video
        out.write(annotated_frame)
        
        # Print progress every 10 frames
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx:04d}: {status_msg}")
            
        if not is_valid:
            if final_result: # Only print and set reason the first time it fails
                final_result = False
                final_reason = status_msg
                print(f"\n[!!!] INSPECTION FAILED AT FRAME {frame_idx} [!!!]")
                print(f"Reason: {status_msg}")
                print("Continuing to process the rest of the video as requested...")
            
        frame_idx += 1
        
    cap.release()
    out.release()
    
    # If the video ended and we are still in Stage 1, it's a FAIL because we never completed the checklist
    if final_result and video_inspector.stage == 1:
        final_result = False
        final_reason = "FAIL: Video ended before YOLO checklist could find all 5 blocks."
        print(f"\n[!!!] INSPECTION FAILED [!!!]")
        print(f"Reason: {final_reason}")
    
    print("\n" + "="*50)
    print("FINAL INSPECTION RESULT:")
    if final_result:
        print("✅ PASS: The remote passed both Stage 1 (YOLO) and Stage 2 (Old Logic).")
    else:
        print(f"❌ FAIL: {final_reason}")
    print("="*50)
    print(f"\nAnnotated video saved to: {Path(args.out).absolute()}")

if __name__ == "__main__":
    main()
