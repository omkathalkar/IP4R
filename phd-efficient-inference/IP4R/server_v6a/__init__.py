"""IP4R v6a — timing-based CNN splash-frame QC pipeline.

Direction B: determine the splash window from video duration, extract the
single best LCD crop from that window, classify it with a fine-tuned
EfficientNet-B0. No phase-detector, no per-frame registration, no LightGBM.
"""
__version__ = "6a.0"
