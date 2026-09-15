from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# 1. Load the model
model_path = r"D:\New Project\IP4R\phd-efficient-inference\IP4R\data\macro_dataset\runs\macro_test\weights\best.pt"
model = YOLO(model_path)

# 2. Setup paths
video_path = r"D:\New Project\Aug-28-2026\12 FPS\NOT GOOD\video@12FPS_20260828_014049.mp4"
out_dir = Path("runs_inference")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "custom_visualization.mp4"

# 3. Define unique colors for our classes (BGR format for OpenCV)
COLORS = {
    0: (255, 0, 0),    # Class 0: Blue
    1: (0, 255, 0),    # Class 1: Green
    2: (0, 0, 255),    # Class 2: Red
    3: (255, 255, 0),  # Class 3: Cyan
    4: (255, 0, 255),  # Class 4: Magenta
    10: (0, 255, 255), # Class 10: Yellow
    11: (255, 128, 0)  # Class 11: Orange
}

print("Processing video with custom ultra-thin boxes and legend...")

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# We will add 200 pixels to the right side of the video for the legend!
legend_width = 200
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(out_path), fourcc, fps, (width + legend_width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run YOLO
    results = model(frame, verbose=False)[0]
    
    # Create a blank legend panel
    legend_panel = np.zeros((height, legend_width, 3), dtype=np.uint8)
    
    # Draw boxes
    counts = {}
    for box in results.boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        
        counts[cls_name] = counts.get(cls_name, 0) + 1
        
        # Get color or default to white
        color = COLORS.get(cls_id, (255, 255, 255))
        
        # Draw ULTRA-THIN box (thickness=1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Draw accuracy/confidence score and class name above the box
        conf = float(box.conf[0])
        label = f"{conf:.2f}"  # Just the accuracy score to keep it clean
        cv2.putText(frame, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw the legend text on the panel
    y_offset = 30
    cv2.putText(legend_panel, "DETECTIONS:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 40
    
    for cls_name, count in counts.items():
        # Find the color for this class name
        cls_id = list(model.names.values()).index(cls_name)
        color = COLORS.get(cls_id, (255, 255, 255))
        
        text = f"{cls_name}: {count}"
        cv2.putText(legend_panel, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y_offset += 30
        
    # Combine the original frame and the legend panel side-by-side
    final_frame = np.hstack((frame, legend_panel))
    
    out.write(final_frame)
    frame_count += 1

cap.release()
out.release()

print("\nDONE!")
print(f"Check this file for the beautiful custom video: {out_path.absolute()}")
