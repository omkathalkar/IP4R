"""Interactive ROI authoring on the golden image (OpenCV HighGUI; works on macOS).

Controls:
  - drag left mouse  : draw a box
  - then type a name in the terminal prompt (blank = discard)
  - u : undo last ROI
  - s : save ROI map to config paths.roi_map
  - q : quit (saves first)
ROIs are stored in normalised coords so they survive resolution changes.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import Config
from ..roi import ROI, load_rois, save_rois


def run_editor(cfg: Config) -> int:
    ref_path = cfg.path("paths.reference_image")
    roi_path = cfg.path("paths.roi_map")
    img = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot read reference: {ref_path}")
        return 1
    H, W = img.shape[:2]
    rois: list[ROI] = load_rois(roi_path)

    state = {"drawing": False, "p0": None, "p1": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["p0"] = (x, y)
            state["p1"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["p1"] = (x, y)
            x0, y0 = state["p0"]
            x1, y1 = state["p1"]
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            if xb - xa < 4 or yb - ya < 4:
                return
            name = input("ROI name (snake_case, blank=discard): ").strip()
            if not name:
                return
            kind = input("kind [icon/label/digit/custom] (default custom): ").strip() or "custom"
            rois.append(ROI(name=name, kind=kind,
                            x=xa / W, y=ya / H, w=(xb - xa) / W, h=(yb - ya) / H))
            print(f"  added {name} ({kind})")

    win = "IP4R ROI editor  [drag=box, u=undo, s=save, q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = img.copy()
        for r in rois:
            x, y, w, h = r.to_pixels(W, H)
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 180, 0), 1)
            cv2.putText(disp, r.name, (x, max(10, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 0), 1, cv2.LINE_AA)
        if state["drawing"] and state["p0"] and state["p1"]:
            cv2.rectangle(disp, state["p0"], state["p1"], (0, 140, 255), 1)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and rois:
            removed = rois.pop()
            print(f"  undo {removed.name}")
        elif key == ord("s"):
            save_rois(rois, roi_path)
            print(f"  saved {len(rois)} ROIs -> {roi_path}")
        elif key in (ord("q"), 27):
            save_rois(rois, roi_path)
            print(f"  saved {len(rois)} ROIs -> {roi_path}")
            break

    cv2.destroyAllWindows()
    return 0
