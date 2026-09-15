import cv2
import sys
from server_v2.lcd_verify import verify_all, draw_cv_overlay

img_path = r"C:\Users\Mohit\.gemini\antigravity-ide\brain\96c7d596-0a3c-4406-a536-c930f8965943\15s_results6\debug_video@12FPS_20260828_014317\04_stage2_cnn_crop_15s.jpg"
img = cv2.imread(img_path)
if img is None:
    print("Could not load image")
    sys.exit(1)

res = verify_all(img)
out = draw_cv_overlay(img, res)
cv2.imwrite(r"scratch/verify_out.jpg", out)
print("Passed:", res['passed'])
