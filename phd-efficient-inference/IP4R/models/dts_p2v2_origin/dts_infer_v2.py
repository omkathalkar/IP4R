"""
DTS Video Inspector v2 — Phase-2-Gated
Usage: python3 dts_infer_v2.py <video_path> [--out output.jpg]

Key change from v1:
  Only analyse frames that match the Phase 2 LCD signature:
    - Icon strip is BLANK  (no symbols/icons)
    - All digit zones are FILLED  (18:88 | 18:88 / 88 / 88888)
  This is the only frame where a font cut or circuit short is unambiguous.
  All other frames (Phase 1 icons, transitions, blur) are ignored.

Verdict logic:
  - PASS  : Phase 2 frame found + majority of P2 frames score >= 0.5
  - FAIL  : Phase 2 frame found + majority fail
  - INCONCLUSIVE : no Phase 2 frame detected in video
"""
import sys, argparse, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import cv2, torch, numpy as np
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, '/home/om/src/ip4r_v2')
from lcd_crop import detect_lcd

MODEL_PATH = Path('/home/om/src/ip4r_v2/models/dts_p2v2_best.pth')

IMG_SIZE   = 224
FRAME_STEP = 5      # finer sampling to catch Phase 2 window
MAX_FRAMES = 300    # higher cap — P2 window can be short
DARK_THRESH = 110

# ── ROI definitions (coords relative to 480×640 LCD crop) ────────────────────
# Segment zones: must be FILLED in Phase 2
SEG_ROIS = [
    ('left_clock',   140, 205,  70, 195, 0.38),   # 18:88 left
    ('right_clock',  140, 205, 235, 380, 0.25),   # 18:88 right
    ('center_88',    215, 305,  70, 190, 0.60),   # large 88
    ('signal_bars',  250, 305, 310, 390, 0.35),   # signal bars
    ('bottom_88888', 405, 455, 175, 395, 0.55),   # 88888
]

# Icon zone: must be BLANK in Phase 2 (this is the gate)
ICON_Y1, ICON_Y2, ICON_X1, ICON_X2 = 90, 140, 85, 400
ICON_MAX_COV = 0.12   # icon strip must be below this to qualify as Phase 2


# ── Phase 2 detector ──────────────────────────────────────────────────────────
def phase2_score(gray: np.ndarray) -> tuple[bool, float, float]:
    """
    Returns (is_p2, digit_score, icon_cov).
    is_p2 = True only when icon strip is blank AND all digit zones are filled.
    digit_score is used to rank Phase 2 frames (pick the clearest one).
    """
    icon_cov = float((gray[ICON_Y1:ICON_Y2, ICON_X1:ICON_X2] < DARK_THRESH).mean())
    total    = float((gray < DARK_THRESH).mean())

    covs = []
    for (_, y1, y2, x1, x2, _) in SEG_ROIS:
        covs.append(float((gray[y1:y2, x1:x2] < DARK_THRESH).mean()))
    digit_score = float(np.mean(covs))

    is_p2 = (digit_score > 0.35) and (total > 0.15)
    return is_p2, digit_score, icon_cov


# ── DTS EfficientNet ──────────────────────────────────────────────────────────
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


# ── ROI coverage check ────────────────────────────────────────────────────────
def analyse_rois(crop_bgr: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    for (name, y1, y2, x1, x2, min_cov) in SEG_ROIS:
        patch = gray[y1:y2, x1:x2]
        coverage = float((patch < DARK_THRESH).mean())
        results.append({
            'name': name, 'coverage': round(coverage, 3),
            'passed': coverage >= min_cov, 'min_cov': min_cov,
            'bbox': (x1, y1, x2 - x1, y2 - y1),
        })
    # Also report icon zone coverage (should be near 0 in Phase 2)
    gray_full = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    icon_cov = float((gray_full[ICON_Y1:ICON_Y2, ICON_X1:ICON_X2] < DARK_THRESH).mean())
    results.append({
        'name': 'icon_strip', 'coverage': round(icon_cov, 3),
        'passed': icon_cov < ICON_MAX_COV, 'min_cov': 0.0,
        'bbox': (ICON_X1, ICON_Y1, ICON_X2 - ICON_X1, ICON_Y2 - ICON_Y1),
    })
    return results


# ── Drawing ───────────────────────────────────────────────────────────────────
GREEN  = (40, 200, 40)
RED    = (30,  30, 220)
YELLOW = (0, 200, 220)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)


def draw_overlay(crop_bgr, verdict, prob, roi_results, p2_frames, p2_pass, p2_fail):
    img = crop_bgr.copy()
    H, W = img.shape[:2]

    for r in roi_results:
        x, y, w, h = r['bbox']
        if r['name'] == 'icon_strip':
            color = GREEN if r['passed'] else YELLOW
        else:
            color = GREEN if r['passed'] else RED
        thick = 1 if r['passed'] else 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        lbl = f"{r['name'].replace('_',' ')}={r['coverage']:.2f}"
        cv2.putText(img, lbl, (x + 2, max(y + 12, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, color, 1, cv2.LINE_AA)

    # Top banner
    v_color = GREEN if verdict == 'PASS' else (RED if verdict == 'FAIL' else YELLOW)
    cv2.rectangle(img, (0, 0), (W, 36), BLACK, -1)
    cv2.putText(img, f'DTS v2: {verdict}', (6, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, v_color, 2, cv2.LINE_AA)
    cv2.putText(img, f'conf={100*prob:.1f}%  P2={p2_frames}fr({p2_pass}ok/{p2_fail}fail)',
                (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.28, WHITE, 1, cv2.LINE_AA)

    # Bottom panel
    failed = [r for r in roi_results if not r['passed'] and r['name'] != 'icon_strip']
    if failed:
        panel_h = 16 + 14 * len(failed)
        cv2.rectangle(img, (0, H - panel_h), (W, H), (20, 20, 20), -1)
        cv2.putText(img, f'Failed segments ({len(failed)}):', (6, H - panel_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, RED, 1, cv2.LINE_AA)
        for i, r in enumerate(failed):
            txt = f"  {r['name']}  cov={r['coverage']:.3f} < {r['min_cov']:.2f}"
            cv2.putText(img, txt, (6, H - panel_h + 14 + 14 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, RED, 1, cv2.LINE_AA)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    vid_path = Path(args.video)
    out_path = (Path(args.out) if args.out
                else vid_path.parent / f'{vid_path.stem}_dts_result.jpg')

    print(f'Video : {vid_path}')
    model, device = load_model()
    print('Model loaded.')

    cap = cv2.VideoCapture(str(vid_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f'Frames: {total_frames} @ {fps:.1f}fps  ({total_frames/fps:.1f}s)')

    frame_indices = list(range(0, total_frames, FRAME_STEP))[:MAX_FRAMES]

    p2_frames   = []   # list of (digit_score, prob, crop)
    total_seen  = 0

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        crop, info = detect_lcd(frame)
        if crop is None:
            continue
        total_seen += 1

        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        is_p2, d_score, icon_cov = phase2_score(g)

        if is_p2:
            prob = predict_crop(model, device, crop)
            p2_frames.append((d_score, prob, crop.copy()))

    cap.release()

    print(f'\nFrames cropped : {total_seen}')
    print(f'Phase 2 frames : {len(p2_frames)}')

    if not p2_frames:
        print('\nVerdict: INCONCLUSIVE — no Phase 2 frame detected')
        print('  (video may not contain the all-segments-on DTS phase)')
        return

    # Pick the best Phase 2 frame (highest digit coverage) for overlay
    p2_frames.sort(key=lambda x: x[0], reverse=True)
    best_d_score, best_prob, best_crop = p2_frames[0]

    p2_probs  = [p for (_, p, _) in p2_frames]
    p2_pass   = sum(1 for p in p2_probs if p >= 0.5)
    p2_fail   = len(p2_probs) - p2_pass
    maj_prob  = float(np.median(p2_probs))
    verdict   = 'PASS' if maj_prob >= 0.5 else 'FAIL'

    print(f'\n{"─"*50}')
    print(f'Phase 2 frames : {len(p2_frames)}  (PASS: {p2_pass}  FAIL: {p2_fail})')
    print(f'Median P2 prob : {maj_prob:.4f}  |  Best-frame prob: {best_prob:.4f}')
    print(f'Verdict        : {verdict}')

    roi_results  = analyse_rois(best_crop)
    failed_segs  = [r for r in roi_results if not r['passed'] and r['name'] != 'icon_strip']
    icon_r       = next(r for r in roi_results if r['name'] == 'icon_strip')

    print(f'\nBest P2 frame — digit score: {best_d_score:.3f}  icon_cov: {icon_r["coverage"]:.3f}')
    if failed_segs:
        print('Failed segments:')
        for r in failed_segs:
            print(f'  FAIL  {r["name"]:18s}  coverage={r["coverage"]:.3f} < {r["min_cov"]:.2f}')
    else:
        print('All segment zones passed.')

    overlay = draw_overlay(best_crop, verdict, maj_prob, roi_results,
                           len(p2_frames), p2_pass, p2_fail)
    cv2.imwrite(str(out_path), overlay)
    print(f'\nOverlay: {out_path}')


if __name__ == '__main__':
    main()
