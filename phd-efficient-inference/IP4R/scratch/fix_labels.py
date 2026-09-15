import os
import re
from pathlib import Path

labels_dir = Path(r"D:\New Project\IP4R\phd-efficient-inference\IP4R\data\yolo_prototype\labels")
images_dir = Path(r"D:\New Project\IP4R\phd-efficient-inference\IP4R\data\yolo_prototype\images")

# Get list of image stems (without extension)
image_stems = [f.stem for f in images_dir.glob("*.jpg")]

print("Fixing Label Studio filename sanitization...")

renamed_count = 0
for txt_file in labels_dir.glob("*.txt"):
    # Label Studio adds a hash like "184b1546-" and removes special chars like "@"
    # Example txt: 184b1546-video12FPS_20260828_013434_sec_0009.txt
    # Expected img: video@12FPS_20260828_013434_sec_0009
    
    # Strip the leading 8-char hash and dash if it exists
    base_txt_name = txt_file.stem
    if re.match(r"^[a-f0-9]{8}-", base_txt_name):
        sanitized_name = base_txt_name[9:] # remove hash and dash
        
        # Now find the matching image stem. We know Label Studio removed the '@'
        matched_stem = None
        for img_stem in image_stems:
            if img_stem.replace("@", "") == sanitized_name:
                matched_stem = img_stem
                break
                
        if matched_stem:
            new_txt_path = labels_dir / f"{matched_stem}.txt"
            os.rename(txt_file, new_txt_path)
            print(f"Renamed: {txt_file.name} -> {new_txt_path.name}")
            renamed_count += 1
        else:
            print(f"Could not find matching image for: {txt_file.name}")

print(f"Successfully renamed {renamed_count} files!")
