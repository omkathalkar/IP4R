"""
DTS Video Inspector
Usage: python3 dts_infer.py <video_path> [--out output.jpg]

Pipeline:
  1. Extract frames from video
  2. LCD-crop each frame via lcd_crop.py
  3. Classify crop with trained DTS EfficientNet-B0
  4. Find the Phase 2 frame with highest confidence to draw annotated output
  5. Overlay ROI coverage map + PASS/FAIL verdict

Outputs:
  - Text verdict (PASS / FAIL + which ROIs are low-coverage)
  - Annotated image showing the failing regions
"""
import sys, argparse, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import cv2, torch, numpy as np
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, '/home/om/src/ip4r_v2')
from lcd_crop import detect_lcd

MODEL_PATH = Path('/home/om/src/ip4r_v2/models/dts_best.pth')

IMG_SIZE   = 224
FRAME_STEP = 10    # sample 1 frame every N frames from the video
MAX_FRAMES = 120   # cap to keep inference fast

# ── DTS EfficientNet ───────────────────────────────────────────────────────────
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 1)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(ckpt['state_dict'])
    m.eval().to(device)
    return m, device

_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_crop(model, device, crop_bgr: np.ndarray) -> float:
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x = _tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze().item()
    return float(torch.sigmoid(torch.tensor(logit)).item())


# ── ROI coverage analysis (for overlay visualisation) ────────────────────────
ROIS = [
    # (name, y1, y2, x1, x2, kind)  kind: expected=must_be_dark, forbidden=must_be_light
    ('left_clock',   140, 205,  70, 195, 'expected',  0.55),
    ('right_clock',  140, 205, 235, 380, 'expected',  0.35),
    ('center_88',    215, 305,  70, 190, 'expected',  0.80),
    ('signal_bars',  250, 305, 310, 390, 'expected',  0.50),
    ('bottom_88888', 405, 455, 175, 395, 'expected',  0.70),
    ('icon_strip',    90, 140,  85, 400, 'expected',  0.40),
]
DARK_THRESH = 110   # pixel darker than this = segment present


def analyse_rois(crop_bgr: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    for (name, y1, y2, x1, x2, kind, min_cov) in ROIS:
        patch = gray[y1:y2, x1:x2]
        coverage = float((patch < DARK_THRESH).mean())
        passed = coverage >= min_cov
        results.append({'name': name, 'coverage': round(coverage, 3),
                        'passed': passed, 'min_cov': min_cov,
                        'bbox': (x1, y1, x2 - x1, y2 - y1)})
    return results


# ── Drawing ───────────────────────────────────────────────────────────────────
GREEN = (40, 200, 40)
RED   = (30,  30, 220)
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)


def draw_overlay(crop_bgr: np.ndarray, prob_pass: float,
                 roi_results: list[dict]) -> np.ndarray:
    img = crop_bgr.copy()
    H, W = img.shape[:2]

    for r in roi_results:
        x, y, w, h = r['bbox']
        color = GREEN if r['passed'] else RED
        thick = 1 if r['passed'] else 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        lbl = f"{r['name']}={r['coverage']:.2f}"
        cv2.putText(img, lbl, (x + 2, max(y + 12, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1, cv2.LINE_AA)

    # Top banner
    verdict = 'PASS' if prob_pass >= 0.5 else 'FAIL'
    v_color = GREEN if prob_pass >= 0.5 else RED
    cv2.rectangle(img, (0, 0), (W, 30), BLACK, -1)
    cv2.putText(img, f'DTS: {verdict}', (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, v_color, 2, cv2.LINE_AA)
    cv2.putText(img, f'conf={100*prob_pass:.1f}%', (W - 140, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1, cv2.LINE_AA)

    # Bottom — failed ROIs
    failed = [r for r in roi_results if not r['passed']]
    if failed:
        panel_h = 16 + 14 * len(failed)
        cv2.rectangle(img, (0, H - panel_h), (W, H), (20, 20, 20), -1)
        cv2.putText(img, 'Low coverage zones:', (6, H - panel_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, RED, 1, cv2.LINE_AA)
        for i, r in enumerate(failed):
            txt = f"  {r['name']}  cov={r['coverage']:.3f} < {r['min_cov']:.2f}"
            cv2.putText(img, txt, (6, H - panel_h + 14 + 14 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, RED, 1, cv2.LINE_AA)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='DTS Video Inspector')
    ap.add_argument('video', help='Path to input video')
    ap.add_argument('--out', default=None, help='Output overlay image path')
    args = ap.parse_args()

    vid_path = Path(args.video)
    out_path = (Path(args.out) if args.out
                else vid_path.parent / f'{vid_path.stem}_dts_result.jpg')

    print(f'Video: {vid_path}')

    model, device = load_model()
    print('Model loaded.')

    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f'Frames: {total_frames} @ {fps:.1f}fps  Duration: {total_frames/fps:.1f}s')

    frame_indices = list(range(0, total_frames, FRAME_STEP))[:MAX_FRAMES]

    best_prob   = 1.0   # worst (most FAIL) frame for overlay
    best_crop   = None
    p2_crop     = None  # best Phase 2 frame (for PASS overlay)
    p2_cov      = -1.0  # combined coverage score for Phase 2 selection
    probs = []

    def _p2_score(gray):
        """Coverage score for Phase 2 detection (high = clearly Phase 2)."""
        c88  = float((gray[215:305, 70:190] < DARK_THRESH).mean())
        bt5  = float((gray[405:455, 175:395] < DARK_THRESH).mean())
        icon = float((gray[90:140, 85:400] < DARK_THRESH).mean())
        lc   = float((gray[140:205, 70:195] < DARK_THRESH).mean())
        return (c88 + bt5 + icon + lc) / 4

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        crop, info = detect_lcd(frame)
        if crop is None:
            continue
        prob = predict_crop(model, device, crop)
        probs.append(prob)

        # Track worst (most FAIL) frame for FAIL overlay
        if prob < best_prob:
            best_prob = prob
            best_crop = crop.copy()

        # Track best Phase 2 frame for PASS overlay
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        score = _p2_score(g)
        if score > p2_cov:
            p2_cov  = score
            p2_crop = crop.copy()

    cap.release()

    if not probs:
        print('No LCD crops found.')
        return

    # Final verdict: FAIL if ANY frame is < 0.5 (worst-case)
    # OR use majority vote
    n_fail = sum(1 for p in probs if p < 0.5)
    n_pass = len(probs) - n_fail
    final_prob = min(probs)   # most pessimistic frame determines verdict
    verdict = 'PASS' if final_prob >= 0.5 else 'FAIL'

    print(f'\n{"─"*50}')
    print(f'Frames analysed: {len(probs)}  (PASS: {n_pass}  FAIL: {n_fail})')
    print(f'Worst-frame prob: {final_prob:.4f}')
    print(f'Verdict: {verdict}')

    # ROI analysis: use Phase 2 frame for PASS, worst frame for FAIL
    analyse_crop = best_crop if verdict == 'FAIL' else (p2_crop if p2_crop is not None else best_crop)
    roi_results = analyse_rois(analyse_crop)
    failed_rois = [r for r in roi_results if not r['passed']]
    if failed_rois:
        print('\nLow-coverage ROIs:')
        for r in failed_rois:
            print(f'  FAIL {r["name"]:18s}  coverage={r["coverage"]:.3f} < {r["min_cov"]:.2f}')
    else:
        print('\nAll ROI zones have normal coverage.')

    overlay = draw_overlay(analyse_crop, final_prob, roi_results)
    cv2.imwrite(str(out_path), overlay)
    print(f'\nOverlay: {out_path}')


if __name__ == '__main__':
    main()
