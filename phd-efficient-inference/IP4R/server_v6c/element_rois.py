"""Per-element pixel ROI definitions for Phase 2A presence checks.

Derived from data/reference/rois.yaml (normalized coords) mapped to the
480×640 canonical LCD crop produced by server_v6a.lcd_crop.detect_lcd().

28 elements covering all icons, digit blocks, and labels.
"""
from __future__ import annotations

from dataclasses import dataclass


LCD_W = 480
LCD_H = 640


@dataclass(frozen=True)
class ElementROI:
    name:  str
    y1:    int   # top    row  (numpy axis 0)
    y2:    int   # bottom row
    x1:    int   # left   col  (numpy axis 1)
    x2:    int   # right  col
    group: str   # broad 8-ROI group this element belongs to


# Pixel coords = round(normalized * LCD_W|H).  Source: data/reference/rois.yaml.
ELEMENTS: list[ElementROI] = [
    # ── Top-row mode icons ────────────────────────────────────────────────────
    ElementROI("auto_mode",        138, 173, 159, 180,  "top_icons"),
    ElementROI("cool_mode",        140, 173, 185, 204,  "top_icons"),
    ElementROI("dry_mode",         142, 173, 207, 220,  "top_icons"),
    ElementROI("fan_mode",         142, 173, 226, 249,  "top_icons"),
    ElementROI("heat_mode",        140, 173, 253, 272,  "top_icons"),
    ElementROI("ir_transmission",  140, 173, 282, 308,  "top_icons"),

    # ── Clock / timer digits ──────────────────────────────────────────────────
    ElementROI("timer_off",        196, 244, 159, 224,  "left_clock"),
    ElementROI("clock",            212, 228, 225, 245,  "left_clock"),
    ElementROI("timer_on",         196, 244, 243, 305,  "right_clock"),

    # ── Timer labels ──────────────────────────────────────────────────────────
    ElementROI("label_time_off",   250, 268, 159, 201,  "left_clock"),
    ElementROI("label_time_on",    250, 268, 251, 295,  "right_clock"),

    # ── Temperature digits & labels ───────────────────────────────────────────
    ElementROI("temperature_tens",  286, 357, 159, 190,  "center_88"),
    ElementROI("temperature_units", 286, 357, 189, 213,  "center_88"),
    ElementROI("label_celsius",     333, 356, 214, 228,  "center_88"),
    ElementROI("label_set_temp",    300, 318, 222, 264,  "center_88"),

    # ── Fan speed ─────────────────────────────────────────────────────────────
    ElementROI("label_auto",       285, 305, 274, 303,  "signal_bars"),
    ElementROI("fan_speed_icon",   306, 329, 276, 291,  "signal_bars"),
    ElementROI("fan_speed_bars",   328, 358, 276, 298,  "signal_bars"),

    # ── Sleep ─────────────────────────────────────────────────────────────────
    ElementROI("sleep_mode",       329, 354, 237, 259,  "secondary_icons"),

    # ── Icon strip: E / Lock / Turbo / ion ───────────────────────────────────
    ElementROI("energy_save_mode", 377, 421, 159, 184,  "icon_strip"),
    ElementROI("lock",             378, 421, 195, 216,  "icon_strip"),
    ElementROI("turbo",            382, 413, 228, 264,  "icon_strip"),
    ElementROI("ion",              382, 421, 278, 304,  "icon_strip"),

    # ── Secondary icons: battery / light / H-swing / V-swing ─────────────────
    ElementROI("battery",  449, 473, 162, 186,  "secondary_icons"),
    ElementROI("light",    446, 477, 203, 223,  "secondary_icons"),
    ElementROI("h_swing",  445, 477, 238, 263,  "secondary_icons"),
    ElementROI("v_swing",  448, 509, 278, 305,  "secondary_icons"),

    # ── Footer digit display ──────────────────────────────────────────────────
    ElementROI("foot_display", 480, 512, 232, 276,  "bottom_88888"),
]

ELEMENT_NAMES: list[str] = [e.name for e in ELEMENTS]
BY_NAME: dict[str, ElementROI] = {e.name: e for e in ELEMENTS}
