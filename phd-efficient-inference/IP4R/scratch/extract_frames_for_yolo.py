import os
import cv2
from pathlib import Path

def extract_frames(video_dir, output_dir):
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all mp4 videos in the folder
    video_files = list(video_dir.glob('*.mp4'))
    if not video_files:
        print(f"No .mp4 files found in {video_dir}")
        return
        
    print(f"Found {len(video_files)} videos. Extracting 1 frame per second...")
    
    for video_path in video_files:
        print(f"Processing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Error opening {video_path.name}")
            continue
            
        # Get the frames per second of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 12.0 # Fallback if OpenCV can't read it
            
        # We want to save exactly 1 frame per second
        frame_interval = int(round(fps))
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # If the current frame is a multiple of the FPS, it's a new second
            if frame_count % frame_interval == 0:
                out_name = f"{video_path.stem}_sec_{saved_count:04d}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), frame)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"  -> Saved {saved_count} frames.")
        
if __name__ == "__main__":
    # Point to the directory with your GOOD videos
    video_dir = r"D:\New Project\Aug-28-2026\12 FPS\GOOD"
    
    # Create a subfolder to store all the images
    output_dir = r"D:\New Project\Aug-28-2026\12 FPS\GOOD\extracted_frames"
    
    extract_frames(video_dir, output_dir)
    print(f"\nFinished successfully! All your images are saved in:")
    print(output_dir)
