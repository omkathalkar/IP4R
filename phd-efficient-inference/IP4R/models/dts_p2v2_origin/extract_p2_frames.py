"""
Phase-2 Frame Extractor
Scans June-27 Good/NotGood DTS videos, LCD-crops each frame,
keeps only frames that look like the Phase 2 splash screen
(18:88 | 18:88 / 88 / 88888, NO icons), and saves them to:
  frames_p2/good/       ← from Good DTS videos
  frames_p2/not_good/   ← from Not Good DTS videos

Phase-2 gate (pixel-level heuristic on the 480×640 crop):
  - bottom_88888 zone dark-pixel coverage  > P2_BT5_MIN
  - center_88 zone dark-pixel coverage     > P2_C88_MIN
  - overall crop dark-pixel fraction       > P2_TOTAL_MIN
  All three must pass — this rejects blank frames, non-DTS phases, bad crops.
"""
import cv2, numpy as np, sys
from pathlib import Path

sys.path.insert(0, '/home/om/src/ip4r_v2')
from lcd_crop import detect_lcd

# ── Paths ──────────────────────────────────────────────────────────────────────
GOOD_DIRS = [
    Path('/home/om/src/FDU Dataset/June-27-2026/Good DTS Videos FHD 1920x1080/12-FPS'),
    Path('/home/om/src/FDU Dataset/June-27-2026/Good DTS Videos FHD 1920x1080/30-FPS'),
]
BAD_DIRS = [
    Path('/home/om/src/FDU Dataset/June-27-2026/Not Good DTS Videos FHD 1920x1080/12-FPS'),
    Path('/home/om/src/FDU Dataset/June-27-2026/Not Good DTS Videos FHD 1920x1080/30-FPS'),
]
OUT_ROOT = Path('/home/om/src/ip4r_v2/frames_p2')

# ── Extraction settings ────────────────────────────────────────────────────────
FRAME_STEP     = 8     # sample every N frames
MAX_PER_VIDEO  = 40    # cap frames per video (balance dataset size vs variety)
DARK_THRESH    = 110

# Phase-2 gate thresholds (tuned from coverage debug on known-good video)
P2_BT5_MIN   = 0.35   # bottom 88888 zone
P2_C88_MIN   = 0.35   # center 88 zone
P2_TOTAL_MIN = 0.15   # whole crop (rejects blank / off-screen frames)

# ROI coords on 480(W) × 640(H) crop
C88_Y1, C88_Y2, C88_X1, C88_X2  = 215, 305,  70, 190
BT5_Y1, BT5_Y2, BT5_X1, BT5_X2  = 405, 455, 175, 395


def is_phase2(gray: np.ndarray) -> bool:
    bt5   = float((gray[BT5_Y1:BT5_Y2, BT5_X1:BT5_X2] < DARK_THRESH).mean())
    c88   = float((gray[C88_Y1:C88_Y2, C88_X1:C88_X2] < DARK_THRESH).mean())
    total = float((gray < DARK_THRESH).mean())
    return bt5 > P2_BT5_MIN and c88 > P2_C88_MIN and total > P2_TOTAL_MIN


def extract_from_dir(video_dirs: list[Path], out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = []
    for d in video_dirs:
        videos.extend(sorted(d.glob('*.mp4')))

    total_saved = 0
    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        indices = list(range(0, n_frames, FRAME_STEP))

        for fi in indices:
            if saved >= MAX_PER_VIDEO:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            crop, info = detect_lcd(frame)
            if crop is None:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if not is_phase2(gray):
                continue
            fname = f'{vid.stem}_f{fi:05d}.jpg'
            cv2.imwrite(str(out_dir / fname), crop)
            saved += 1

        cap.release()
        total_saved += saved
        print(f'  [{label}] {vid.name}: {saved} frames saved')

    print(f'  → Total {label}: {total_saved} frames\n')
    return total_saved


if __name__ == '__main__':
    print('=== Phase-2 Frame Extraction ===\n')
    n_good = extract_from_dir(GOOD_DIRS,    OUT_ROOT / 'good',     'good')
    n_bad  = extract_from_dir(BAD_DIRS,     OUT_ROOT / 'not_good', 'not_good')
    print(f'Dataset ready:  good={n_good}  not_good={n_bad}  total={n_good+n_bad}')
    print(f'Output: {OUT_ROOT}')
