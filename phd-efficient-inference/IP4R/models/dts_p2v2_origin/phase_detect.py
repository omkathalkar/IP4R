"""
Phase Detector — classifies a cropped LCD image into one of:
  'blank'   — dark screen, nothing lit
  'phase1'  — partial DTS pattern (18:88 | 18:88 | 88 | bars | 88888)
  'phase2'  — all segments on (very bright, high pixel density)

Strategy (rule-based, no training needed):
  1. Convert crop to grayscale
  2. Measure mean brightness of the entire crop
  3. Measure pixel density in known segment zones
     - Digit zone (center strip): Phase 1 and Phase 2 both bright here
     - Icon zone (top ~15% of LCD): Phase 2 lights up many small icons;
       Phase 1 has only a few.
  4. Decision tree:
       mean < BLANK_THRESH              → blank
       icon_density > PHASE2_ICON_THRESH → phase2
       else                              → phase1

LCD_W=480, LCD_H=640 (from lcd_crop.py)
Zones are fractions of LCD_H / LCD_W.
"""
import cv2
import numpy as np
from pathlib import Path

LCD_W, LCD_H = 480, 640

# Brightness thresholds (0-255 scale)
BLANK_THRESH        = 15     # mean brightness below this → blank
PHASE2_ICON_THRESH  = 0.30   # icon zone lit fraction above this → phase2

# Icon zone: top 12% of LCD height, full width
ICON_Y1 = 0
ICON_Y2 = int(LCD_H * 0.12)   # ~77 px

# Digit zone: middle 60% of height, central 80% of width
DIGIT_Y1 = int(LCD_H * 0.18)
DIGIT_Y2 = int(LCD_H * 0.78)
DIGIT_X1 = int(LCD_W * 0.05)
DIGIT_X2 = int(LCD_W * 0.95)

# Pixel brightness threshold to count as "lit"
LIT_THRESH = 80


def classify_phase(crop_bgr: np.ndarray) -> dict:
    """
    Returns dict:
      phase: 'blank' | 'phase1' | 'phase2'
      mean_brightness: float
      icon_density: float  (fraction of lit pixels in icon zone)
      digit_density: float (fraction of lit pixels in digit zone)
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())

    if mean_brightness < BLANK_THRESH:
        return {'phase': 'blank', 'mean_brightness': mean_brightness,
                'icon_density': 0.0, 'digit_density': 0.0}

    icon_patch  = gray[ICON_Y1:ICON_Y2, :]
    digit_patch = gray[DIGIT_Y1:DIGIT_Y2, DIGIT_X1:DIGIT_X2]

    icon_density  = float((icon_patch  > LIT_THRESH).mean())
    digit_density = float((digit_patch > LIT_THRESH).mean())

    phase = 'phase2' if icon_density > PHASE2_ICON_THRESH else 'phase1'

    return {
        'phase': phase,
        'mean_brightness': mean_brightness,
        'icon_density': icon_density,
        'digit_density': digit_density,
    }


def classify_from_path(img_path: str | Path) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        return {'phase': 'error', 'reason': 'read_failed'}
    return classify_phase(img)


# ── CLI: scan a folder and report distribution ─────────────────────────────────
if __name__ == '__main__':
    import sys, json
    from collections import Counter

    if len(sys.argv) < 2:
        print('Usage: python3 phase_detect.py <folder_of_crops> [--save-json out.json]')
        sys.exit(1)

    folder = Path(sys.argv[1])
    save_json = None
    if '--save-json' in sys.argv:
        idx = sys.argv.index('--save-json')
        save_json = sys.argv[idx + 1]

    crops = sorted(folder.glob('*.jpg'))
    print(f'Classifying {len(crops)} crops in {folder}...')

    results = []
    counts = Counter()
    for p in crops:
        r = classify_from_path(p)
        r['file'] = p.name
        results.append(r)
        counts[r['phase']] += 1

    print(f'\nPhase distribution:')
    for phase, n in sorted(counts.items()):
        print(f'  {phase:12s}: {n:5d}  ({100*n/len(crops):.1f}%)')

    # Print a few samples per phase for sanity check
    print('\nSample stats per phase:')
    for phase in ('blank', 'phase1', 'phase2'):
        subset = [r for r in results if r['phase'] == phase][:3]
        for r in subset:
            print(f"  [{phase}] {r['file']}  mean={r['mean_brightness']:.1f}"
                  f"  icon={r['icon_density']:.3f}  digit={r['digit_density']:.3f}")

    if save_json:
        with open(save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved: {save_json}')
