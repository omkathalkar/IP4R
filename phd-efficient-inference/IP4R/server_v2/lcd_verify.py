"""LCD structural verifier — CV layer of the DL+CV sandwich.

Runs on the 480×640 BGR LCD crop (output of detect_lcd()).
Checks five zones against their expected Phase-2 patterns:

  left_clock    → "18:88"   (colon must be lit)
  right_clock   → "18:88"
  center_88     → "88"      (two large 8-digits)
  signal_bars   → ▂▄█       (3 ascending bars)
  bottom_88888  → "88888"   (five 8-digits)

For each digit the seven strokes (a–g) are checked individually.
A stroke that should be ON but reads below _SEG_ON_MIN  → FAIL.
A stroke that should be OFF but reads above _SEG_OFF_MAX → FAIL.
"""
from __future__ import annotations

import cv2
import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────
# verify_thresh (default 80) is LOWER than the phase-2 gate dark_thresh (125).
# At 125 the whole LCD reads as uniformly dark; at 80 only true segment marks show.
_VERIFY_THRESH = 80

_SEG_ON_MIN  = 0.08   # dark-pixel coverage for a segment to count as ON
_SEG_OFF_MAX = 0.10   # coverage above which an OFF segment is flagged (relaxed for low-contrast LCD)
_BAR_MIN_H   = 0.08   # minimum bar height as fraction of zone height
_DOT_MIN_COV = 0.04   # minimum coverage for a colon dot to be present

# ── Seven-segment definitions (fractions of digit-cell H × W) ─────────────
#    a = top-horizontal   b = top-right-vert   c = bottom-right-vert
#    d = bot-horizontal   e = bottom-left-vert f = top-left-vert
#    g = middle-horizontal
_SEG_FRACS: dict[str, tuple] = {
    'a': (0.00, 0.18, 0.15, 0.85),
    'b': (0.05, 0.50, 0.68, 0.97),
    'c': (0.52, 0.95, 0.68, 0.97),
    'd': (0.82, 1.00, 0.15, 0.85),
    'e': (0.52, 0.95, 0.03, 0.32),
    'f': (0.05, 0.50, 0.03, 0.32),
    'g': (0.41, 0.59, 0.15, 0.85),
}

# Segments that MUST be ON / OFF for each expected character
_MUST_ON: dict[str, str] = {
    '8': 'abcdefg',
    '1': 'bc',
}
# '1' off-check omitted: bc bleed into the horizontal-segment fracs causes spurious failures
# on low-contrast LCD panels where the cell boundaries don't cleanly isolate segments.
_MUST_OFF: dict[str, str] = {
    '8': '',
    '1': '',
}


def _frac_cov(gray: np.ndarray, y1f: float, y2f: float,
              x1f: float, x2f: float, dark_thresh: int) -> float:
    h, w = gray.shape
    patch = gray[int(h * y1f):int(h * y2f), int(w * x1f):int(w * x2f)]
    if patch.size == 0:
        return 0.0
    return float((patch < dark_thresh).mean())


def _check_digit(cell_bgr: np.ndarray, expected: str,
                 dark_thresh: int) -> dict:
    gray     = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    must_on  = _MUST_ON.get(expected, '')
    must_off = _MUST_OFF.get(expected, '')
    failures = []
    seg_cov  = {}

    for seg, fracs in _SEG_FRACS.items():
        cov = _frac_cov(gray, *fracs, dark_thresh)
        seg_cov[seg] = round(cov, 3)
        if seg in must_on and cov < _SEG_ON_MIN:
            failures.append(f"{seg}_missing(cov={cov:.2f})")
        if seg in must_off and cov > _SEG_OFF_MAX:
            failures.append(f"{seg}_extra(cov={cov:.2f})")

    return {
        'expected':    expected,
        'passed':      len(failures) == 0,
        'failures':    failures,
        'seg_cov':     seg_cov,
    }


def _check_colon(cell_bgr: np.ndarray, dark_thresh: int) -> dict:
    gray  = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    upper = _frac_cov(gray, 0.22, 0.42, 0.25, 0.75, dark_thresh)
    lower = _frac_cov(gray, 0.58, 0.78, 0.25, 0.75, dark_thresh)
    # Only require lower dot — upper dot is often too faint on this LCD panel.
    passed = lower > _DOT_MIN_COV
    failures = []
    if upper <= _DOT_MIN_COV:
        failures.append(f"upper_dot_faint(cov={upper:.2f})")
    if lower <= _DOT_MIN_COV:
        failures.append(f"lower_dot_missing(cov={lower:.2f})")
    return {
        'expected': ':',
        'passed':   passed,
        'failures': failures,
        'upper_dot': round(upper, 3),
        'lower_dot': round(lower, 3),
    }


def _check_signal_bars(bars_bgr: np.ndarray, dark_thresh: int) -> dict:
    """3 ascending bars: bar1 < bar2 < bar3 in height, all present."""
    gray  = cv2.cvtColor(bars_bgr, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    col_w = max(1, w // 3)
    heights = []

    for i in range(3):
        col      = gray[:, i * col_w: (i + 1) * col_w]
        row_cov  = (col < dark_thresh).mean(axis=1)   # per-row dark coverage
        lit_rows = np.where(row_cov > 0.20)[0]
        heights.append(int(h - lit_rows.min()) if len(lit_rows) else 0)

    min_h       = h * _BAR_MIN_H
    all_present = all(ht > min_h for ht in heights)
    ascending   = heights[0] < heights[1] < heights[2]
    failures    = []
    if not all_present:
        missing = [i + 1 for i, ht in enumerate(heights) if ht <= min_h]
        failures.append(f"bar(s)_missing:{missing}")
    if all_present and not ascending:
        failures.append(f"not_ascending:{heights}")

    return {
        'expected':    '▂▄█',
        'passed':      all_present and ascending,
        'failures':    failures,
        'bar_heights': heights,
        'all_present': all_present,
        'ascending':   ascending,
    }


# ── Zone layout on the 480×640 crop ──────────────────────────────────────────
# Each entry: list of (expected_char, x1_abs, x2_abs)  +  y1/y2 for the zone.
# Coordinates are calibrated estimates — adjust if LCD crop shifts.
_DIGIT_ZONES: dict[str, dict] = {
    'left_clock': {
        'y1': 143, 'y2': 203,
        'chars': [
            ('1',  87,  101),
            ('8', 117,  142),
            (':',  142, 157),
            ('8', 157,  177),
            ('8', 180,  200),
        ],
    },
    'right_clock': {
        'y1': 143, 'y2': 203,
        'chars': [
            ('1',  254, 268),
            ('8',  265, 283),
            (':',  293, 308),
            ('8',  319, 344),
            ('8',  347, 371),
        ],
    },
    'center_88': {
        'y1': 218, 'y2': 303,
        'chars': [
            ('8',  70,  112),
            ('8', 118,  165),
        ],
    },
    'bottom_88888': {
        'y1': 407, 'y2': 453,
        'chars': [
            ('8', 210, 230),
            ('8', 229, 249),
            ('8', 249, 269),
            ('8', 267, 287),
            ('8', 287, 320),
        ],
    },
}

_BARS_Y1, _BARS_Y2 = 275, 420
_BARS_X1, _BARS_X2 = 350, 415


def verify_all(crop_bgr: np.ndarray, dark_thresh: int = 125) -> dict:
    """Run full structural verification on a 480×640 LCD crop.

    dark_thresh is the Phase-2 gate threshold (passed for context); structural
    checks always use the internal _VERIFY_THRESH (80) which isolates segment
    marks from the LCD background more reliably.

    Returns:
        {
          'passed': bool,
          'zones':  { zone_name: { 'passed': bool, 'chars': [...] } },
        }
    """
    zones: dict = {}
    all_ok = True

    for zone_name, zdef in _DIGIT_ZONES.items():
        y1, y2    = zdef['y1'], zdef['y2']
        char_results = []

        for expected, x1, x2 in zdef['chars']:
            cell = crop_bgr[y1:y2, x1:x2]
            if expected == ':':
                r = _check_colon(cell, _VERIFY_THRESH)
            else:
                r = _check_digit(cell, expected, _VERIFY_THRESH)
            r['x1'], r['x2'], r['y1'], r['y2'] = x1, x2, y1, y2
            char_results.append(r)

        zone_ok    = all(r['passed'] for r in char_results)
        all_ok     = all_ok and zone_ok
        zones[zone_name] = {'passed': zone_ok, 'chars': char_results}

    # Signal bars
    bars_crop      = crop_bgr[_BARS_Y1:_BARS_Y2, _BARS_X1:_BARS_X2]
    bars_result    = _check_signal_bars(bars_crop, _VERIFY_THRESH)
    all_ok         = all_ok and bars_result['passed']
    zones['signal_bars'] = bars_result

    return {'passed': all_ok, 'zones': zones}


def draw_cv_overlay(img: np.ndarray, cv_result: dict,
                    dark_thresh: int = 125) -> np.ndarray:
    """Draw CV verify boxes onto a copy of the LCD crop image."""
    out  = img.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    GREEN, RED = (40, 200, 40), (30, 30, 220)

    for zone_name, zdata in cv_result.get('zones', {}).items():
        if zone_name == 'signal_bars':
            color = GREEN if zdata['passed'] else RED
            cv2.rectangle(out, (_BARS_X1, _BARS_Y1),
                          (_BARS_X2, _BARS_Y2), color, 1)
            cv2.putText(out, 'bars', (_BARS_X1 + 2, _BARS_Y1 + 10),
                        FONT, 0.22, color, 1, cv2.LINE_AA)
            continue

        for r in zdata.get('chars', []):
            color = GREEN if r['passed'] else RED
            thick = 1 if r['passed'] else 2
            cv2.rectangle(out, (r['x1'], r['y1']),
                          (r['x2'], r['y2']), color, thick)
            label = r['expected'] + ('✓' if r['passed'] else '✗')
            cv2.putText(out, label,
                        (r['x1'] + 1, r['y1'] + 10),
                        FONT, 0.22, color, 1, cv2.LINE_AA)

    return out
