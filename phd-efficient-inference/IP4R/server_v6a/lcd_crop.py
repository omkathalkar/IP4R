"""LCD perspective-crop — lifted verbatim from server_v3 so v6a has no dependency on it."""
from __future__ import annotations

import cv2
import numpy as np

LCD_W, LCD_H = 480, 640


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def _perspective_crop(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [LCD_W, 0], [LCD_W, LCD_H], [0, LCD_H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (LCD_W, LCD_H))


def detect_lcd(frame: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Return (480×640 BGR crop, info_dict). crop is None on failure."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Pass 1: LCD glass is a dark inner hole in the white remote body.
    _, white_thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    k30 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    white_closed = cv2.morphologyEx(white_thresh, cv2.MORPH_CLOSE, k30)
    cnts, hier = cv2.findContours(white_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cnts and hier is not None:
        inner = sorted(
            [(cv2.contourArea(c), c) for c, h2 in zip(cnts, hier[0])
             if h2[3] != -1 and h * w * 0.03 < cv2.contourArea(c) < h * w * 0.40],
            key=lambda x: x[0], reverse=True,
        )
        for area, cnt in inner:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.08 * peri, True)
            if len(approx) == 4:
                corners = _order_corners(approx)
                rect_area = cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.int32))
                if h * w * 0.03 < rect_area < h * w * 0.40:
                    return _perspective_crop(frame, corners), {
                        "success": True, "method": "perspective_inner",
                        "corners": corners.tolist(),
                    }

    # Pass 2: Otsu threshold + outer-contour.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k15)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"success": False, "method": "none"}

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        if cv2.contourArea(cnt) < h * w * 0.05:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            corners = _order_corners(approx)
            rect_area = cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.int32))
            if h * w * 0.05 < rect_area < h * w * 0.60:
                return _perspective_crop(frame, corners), {
                    "success": True, "method": "perspective",
                    "corners": corners.tolist(),
                }

    # Fallback: bounding box.
    x, y, bw, bh = cv2.boundingRect(contours[0])
    if bw * bh > h * w * 0.60:
        return None, {"success": False, "method": "bbox_too_large"}
    pad = 5
    x, y = max(0, x - pad), max(0, y - pad)
    bw, bh = min(w - x, bw + 2 * pad), min(h - y, bh + 2 * pad)
    crop = cv2.resize(frame[y:y + bh, x:x + bw], (LCD_W, LCD_H))
    return crop, {"success": True, "method": "bbox", "bbox": [x, y, bw, bh]}
