"""
Batch evaluate DTS model on a sample of good and not-good videos.
Prints per-video verdict and final accuracy.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/om/src/ip4r_v2')

import cv2, torch, random
from torchvision import transforms, models
from pathlib import Path
from PIL import Image

MODEL_PATH = Path('/home/om/src/ip4r_v2/models/dts_best.pth')
GOOD_DIR   = Path('/home/om/src/FDU Dataset/June-27-2026/Good DTS Videos FHD 1920x1080')
NG_DIR     = Path('/home/om/src/FDU Dataset/June-27-2026/Not Good DTS Videos FHD 1920x1080')

FRAME_STEP = 15
MAX_FRAMES = 60

from lcd_crop import detect_lcd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = models.efficientnet_b0(weights=None)
m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 1)
ckpt = torch.load(MODEL_PATH, map_location=device)
m.load_state_dict(ckpt['state_dict'])
m.eval().to(device)

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def infer_video(vid_path, expected_pass: bool):
    cap = cv2.VideoCapture(str(vid_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, FRAME_STEP))[:MAX_FRAMES]
    probs = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: continue
        crop, info = detect_lcd(frame)
        if crop is None: continue
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = m(x).squeeze().item()
        probs.append(float(torch.sigmoid(torch.tensor(logit)).item()))
    cap.release()
    if not probs:
        return None, 'no_crop'
    worst = min(probs)
    verdict = 'PASS' if worst >= 0.5 else 'FAIL'
    correct = (verdict == 'PASS') == expected_pass
    return correct, f'{verdict} (worst={worst:.3f}, n={len(probs)})'


good_videos = sorted(GOOD_DIR.rglob('*.mp4'))
ng_videos   = sorted(NG_DIR.rglob('*.mp4'))

# Sample 20 from each for speed
random.seed(0)
good_sample = random.sample(good_videos, min(20, len(good_videos)))
ng_sample   = random.sample(ng_videos,   min(20, len(ng_videos)))

results = []
print('─' * 65)
print(f'{"Video":<40}  {"Expected":<8}  {"Result"}')
print('─' * 65)
for (vids, expected_pass, tag) in [(good_sample, True, 'GOOD'), (ng_sample, False, 'NG')]:
    for vp in vids:
        correct, msg = infer_video(vp, expected_pass)
        status = '✓' if correct else '✗'
        print(f'{status} [{tag}] {vp.name[:38]:38s}  {msg}')
        results.append(correct)

acc = sum(results) / len(results)
print('─' * 65)
print(f'Accuracy: {sum(results)}/{len(results)} = {acc*100:.1f}%')
