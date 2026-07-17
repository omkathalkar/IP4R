"""LCD Auto-Crop — perspective-corrects the LCD panel from a raw video frame.

Returns a fixed 480×640 BGR crop. All downstream ROI coordinates are relative
to this canvas size.
"""
from __future__ import annotations

import cv2
import numpy as np

LCD_W, LCD_H = 480, 640


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],   # top-left
        pts[np.argmin(d)],   # top-right
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left
    ], dtype=np.float32)


def _perspective_crop(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [LCD_W, 0], [LCD_W, LCD_H], [0, LCD_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (LCD_W, LCD_H))


def detect_lcd(frame: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Return (cropped_lcd, info_dict). crop is LCD_W×LCD_H BGR, or None on failure."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"success": False, "method": "none"}

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        if cv2.contourArea(cnt) < h * w * 0.05:
            break
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            corners   = _order_corners(approx)
            rect_area = cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.int32))
            if rect_area > h * w * 0.05:
                return _perspective_crop(frame, corners), {
                    "success": True, "method": "perspective",
                    "corners": corners.tolist(),
                }

    # Fallback: bounding box of largest contour
    x, y, bw, bh = cv2.boundingRect(contours[0])
    pad = 5
    x, y  = max(0, x - pad), max(0, y - pad)
    bw, bh = min(w - x, bw + 2 * pad), min(h - y, bh + 2 * pad)
    crop = cv2.resize(frame[y:y + bh, x:x + bw], (LCD_W, LCD_H))
    return crop, {"success": True, "method": "bbox", "bbox": [x, y, bw, bh]}
