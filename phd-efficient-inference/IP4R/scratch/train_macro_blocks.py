import os
import re
from pathlib import Path
from ultralytics import YOLO

def main():
    dest_dir = Path(r"D:\New Project\IP4R\phd-efficient-inference\IP4R\data\macro_dataset")
    yaml_path = dest_dir / "dataset.yaml"
    
    # Fix Label Studio Filenames
    print("Fixing Label Studio file names...")
    labels_dir = dest_dir / "labels"
    images_dir = dest_dir / "images"
    image_stems = [f.stem for f in images_dir.glob("*.jpg")]
    
    for txt_file in labels_dir.glob("*.txt"):
        base_txt_name = txt_file.stem
        # Remove the 8-character hash prefix if present
        if re.match(r"^[a-f0-9]{8}-", base_txt_name):
            sanitized_name = base_txt_name[9:]
            # Find the corresponding image stem
            for img_stem in image_stems:
                if img_stem.replace("@", "") == sanitized_name:
                    new_path = labels_dir / f"{img_stem}.txt"
                    os.rename(txt_file, new_path)
                    break
        else:
            sanitized_name = base_txt_name
            for img_stem in image_stems:
                if img_stem.replace("@", "") == sanitized_name:
                    new_path = labels_dir / f"{img_stem}.txt"
                    if str(txt_file) != str(new_path):
                        os.rename(txt_file, new_path)
                    break
                    
    print("\nStarting YOLO training on MACRO BLOCKS...")
    model = YOLO('yolov8n.pt') 
    
    results = model.train(
        data=str(yaml_path), 
        epochs=300,  # 300 epochs should be enough for large macro blocks
        imgsz=640, 
        project=str(dest_dir / "runs"),
        name="macro_test"
    )
    
    print(f"\n✅ Training complete!")
    print(f"Your trained model file is saved at: {dest_dir / 'runs' / 'macro_test' / 'weights' / 'best.pt'}")

if __name__ == "__main__":
    main()
