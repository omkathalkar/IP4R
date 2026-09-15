import os
import shutil
import yaml
from pathlib import Path

# We import YOLO to run the training
try:
    from ultralytics import YOLO
except ImportError:
    print("YOLO is not installed yet! Waiting for pip install to finish...")
    exit(1)

# 1. Define Paths
source_dir = Path(r"C:\Users\Mohit\Downloads\project-4-at-2026-09-14-17-22-0fb7e6be")
dest_dir = Path(r"D:\New Project\IP4R\phd-efficient-inference\IP4R\data\yolo_prototype")

# 2. Copy Data to Workspace (so it's not sitting in your Downloads folder)
print("1. Moving your dataset into the project folder...")
if dest_dir.exists():
    shutil.rmtree(dest_dir)
shutil.copytree(source_dir, dest_dir)
print(f"   -> Copied to {dest_dir}")

# 3. Create dataset.yaml
print("2. Generating dataset.yaml...")
yaml_path = dest_dir / "dataset.yaml"
data = {
    "path": str(dest_dir.absolute()), # Tell YOLO where the root folder is
    "train": "images",                # Tell YOLO where train images are
    "val": "images",                  # For this tiny prototype, we validate on the same images
    "names": {}
}

import re

# Automatically read the classes.txt that Label Studio generated!
classes_file = dest_dir / "classes.txt"
if classes_file.exists():
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    data["names"] = {i: name for i, name in enumerate(class_names)}
else:
    print("   -> ERROR: classes.txt not found. Cannot create yaml!")
    exit(1)

# Fix Label Studio Filenames
print("2.5 Fixing Label Studio file names...")
labels_dir = dest_dir / "labels"
images_dir = dest_dir / "images"
image_stems = [f.stem for f in images_dir.glob("*.jpg")]
for txt_file in labels_dir.glob("*.txt"):
    base_txt_name = txt_file.stem
    if re.match(r"^[a-f0-9]{8}-", base_txt_name):
        sanitized_name = base_txt_name[9:]
        for img_stem in image_stems:
            if img_stem.replace("@", "") == sanitized_name:
                os.rename(txt_file, labels_dir / f"{img_stem}.txt")
                break

# Save the yaml file
with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"   -> Created {yaml_path}")

# 4. Train the Model!
print("\n3. Starting YOLO training (this will download the base AI brain and train it on your 19 images)...")
model = YOLO('yolov8n.pt') # Load the nano model (fastest)

# Run the training for 800 epochs (passes over the data)
results = model.train(
    data=str(yaml_path), 
    epochs=800, 
    imgsz=640, 
    project=str(dest_dir / "runs"),
    name="startup_test"
)

print(f"\n✅ Training complete! The AI is now trained.")
print(f"Your trained model file is saved at:")
print(dest_dir / "runs" / "startup_test" / "weights" / "best.pt")
