"""Phase 2B — AnomalyDINO whole-crop catch-all (DINOv2 nearest-neighbor anomaly).

Catches defects that escape the per-element atlas in phase2_elements.py (Part A):
cracks, wide-area dimming, wrong overall LCD content, ghosting, or any novel
pattern not reducible to "one named element is missing."

Algorithm (one video):
  1. Load DINOv2 (facebook/dinov2-small) — no fine-tuning, inference only.
  2. Load reference bank (.npz) built from CALIBRATION GOOD videos.
  3. Extract N_TIMEPOINTS consecutive LCD crops from T*.
  4. For each crop: DINOv2 patch tokens → cosine NN distance to bank per live
     patch → image score = MAX over patches (PatchCore/AnomalyDINO convention).
  5. Persistence vote: FAIL if >= CONFIRM_HITS frames exceed threshold.

IMPORTANT: build reference bank and calibrate threshold using ONLY
           Jul-14 GOOD (5 vids) + Sep-09 GOOD (13 vids).
           Sep-15 is the locked eval set — never use during calibration.

Build reference bank:
    python -m server_v6c.phase2_dino build-bank \\
        --dataset '/path/Jul-14/GOOD,/path/Sep-09/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --out models/dino_reference_bank.npz

Calibrate threshold (leave-one-out on calibration GOOD):
    python -m server_v6c.phase2_dino calibrate \\
        --dataset '/path/Jul-14/GOOD,/path/Sep-09/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --out models/dino_threshold.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

DINO_MODEL_ID  = "facebook/dinov2-small"
N_TIMEPOINTS   = 5
CONFIRM_HITS   = 4     # >= this many frames above threshold → FAIL
CAL_PERCENTILE = 97.0  # percentile of LOO GOOD scores → threshold


# ---------------------------------------------------------------------------
# Module-level caches — avoid reloading model / bank on each call
# ---------------------------------------------------------------------------

_DINO_CACHE:      dict[tuple, object]    = {}   # (model_id, device) → (processor, model)
_BANK_NORM_CACHE: dict[str, np.ndarray] = {}   # bank_path str → pre-normalised matrix


def _warn_if_eval_data(dirs: list[Path]) -> None:
    for d in dirs:
        if any(s in str(d) for s in ("Sep-15", "Sep15", "Sep_15")):
            log.warning("!" * 70)
            log.warning("LEAKAGE WARNING: '%s' looks like Sep-15 (locked eval set).", d)
            log.warning("Calibration must use ONLY Jul-14 GOOD + Sep-09 GOOD (18 videos).")
            log.warning("!" * 70)


def _get_dino(model_id: str = DINO_MODEL_ID, device: str = "cpu"):
    key = (model_id, device)
    if key not in _DINO_CACHE:
        log.info("Loading DINOv2 %s on %s …", model_id, device)
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).eval().to(device)
        _DINO_CACHE[key] = (processor, model)
    return _DINO_CACHE[key]


def _get_bank_norm(bank_path: Path) -> np.ndarray:
    key = str(bank_path)
    if key not in _BANK_NORM_CACHE:
        data = np.load(str(bank_path), allow_pickle=True)
        bank = data["bank"].astype(np.float32)
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
        _BANK_NORM_CACHE[key] = bank / (norms + 1e-8)
    return _BANK_NORM_CACHE[key]


# ---------------------------------------------------------------------------
# DINOv2 feature extraction
# ---------------------------------------------------------------------------

def _extract_patch_tokens(
    crop_bgr: np.ndarray,
    processor,
    model,
    device: str = "cpu",
) -> np.ndarray:
    """Return (num_patches, dim) float32 patch tokens for one BGR crop."""
    import torch  # noqa: PLC0415
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # last_hidden_state: (1, 1+num_patches, dim) — index 0 is CLS, drop it
    return out.last_hidden_state[0, 1:, :].cpu().numpy().astype(np.float32)


def _anomaly_score_from_norm(live_tokens: np.ndarray, ref_norm: np.ndarray) -> float:
    """Cosine NN anomaly: image score = max per-patch distance to nearest ref patch."""
    live_norm = live_tokens / (np.linalg.norm(live_tokens, axis=1, keepdims=True) + 1e-8)
    sims = live_norm @ ref_norm.T               # (P_live, P_ref)
    nearest_sim = sims.max(axis=1)              # best match per live patch
    patch_dist = 1.0 - nearest_sim.clip(0.0, 1.0)
    return float(patch_dist.max())              # PatchCore convention: MAX, not mean


# ---------------------------------------------------------------------------
# Reference bank builder
# ---------------------------------------------------------------------------

def build_reference_bank(
    dataset_dirs:    list[Path],
    yolo_model_path: Path,
    out_path:        Path,
    model_id:        str = DINO_MODEL_ID,
    n_samples:       int = 20,
    device:          str = "cpu",
) -> Path:
    """Build patch-token reference bank from confirmed-GOOD calibration videos.

    IMPORTANT: dataset_dirs must contain ONLY Jul-14 GOOD + Sep-09 GOOD.
               NEVER include Sep-15 — it is the locked eval set.
    """
    from .yolo_phase1 import run_yolo_phase1
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    _warn_if_eval_data(dataset_dirs)
    processor, model = _get_dino(model_id, device)

    videos: list[Path] = []
    for d in dataset_dirs:
        videos.extend(sorted(p for p in Path(d).iterdir() if p.suffix.lower() in _VIDEO_EXTS))
    if not videos:
        raise RuntimeError(f"No video files found in {dataset_dirs}")

    log.info("Building DINOv2 reference bank: %d/%d videos", min(n_samples, len(videos)), len(videos))

    all_patches: list[np.ndarray] = []
    used = 0
    for vid in videos[:n_samples]:
        log.info("  %s …", vid.name)
        p1 = run_yolo_phase1(vid, yolo_model_path)
        if not p1.complete or p1.T_star is None:
            log.warning("    Phase1 FAIL — skipping")
            continue
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, p1.T_star)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            log.warning("    Cannot read frame %d — skipping", p1.T_star)
            continue
        crop, _ = detect_lcd(frame)
        if crop is None:
            log.warning("    detect_lcd failed at T*=%d — skipping", p1.T_star)
            continue
        tokens = _extract_patch_tokens(crop, processor, model, device)
        all_patches.append(tokens)
        log.info("    OK  T*=%d (%.1fs)  shape=%s", p1.T_star, p1.T_star_sec, tokens.shape)
        used += 1

    if used == 0:
        raise RuntimeError("No valid samples — cannot build reference bank")

    bank = np.concatenate(all_patches, axis=0).astype(np.float32)
    log.info("Reference bank: %s  (%d videos)", bank.shape, used)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        bank     = bank,
        n_videos = np.array([used]),
        model_id = np.array([model_id], dtype=object),
    )
    log.info("Reference bank saved → %s", out_path)
    _BANK_NORM_CACHE.pop(str(out_path), None)   # invalidate cache after rebuild
    return out_path


# ---------------------------------------------------------------------------
# LOO threshold calibration
# ---------------------------------------------------------------------------

def calibrate_dino_threshold(
    dataset_dirs:    list[Path],
    yolo_model_path: Path,
    out_path:        Path,
    model_id:        str   = DINO_MODEL_ID,
    n_samples:       int   = 20,
    cal_pct:         float = CAL_PERCENTILE,
    device:          str   = "cpu",
) -> float:
    """Leave-one-out calibration of the anomaly threshold.

    For each video: build a temporary bank from all others, score the held-out
    video. Threshold = cal_pct-th percentile of LOO scores.

    IMPORTANT: use ONLY Jul-14 GOOD + Sep-09 GOOD. Never include Sep-15.
    """
    from .yolo_phase1 import run_yolo_phase1
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    _warn_if_eval_data(dataset_dirs)
    processor, dino_model = _get_dino(model_id, device)

    videos: list[Path] = []
    for d in dataset_dirs:
        videos.extend(sorted(p for p in Path(d).iterdir() if p.suffix.lower() in _VIDEO_EXTS))
    if len(videos) < 2:
        raise RuntimeError("Need at least 2 calibration videos for LOO")

    videos = videos[:n_samples]
    log.info("Pre-extracting DINOv2 tokens for %d calibration videos …", len(videos))

    token_list: list[tuple[str, np.ndarray | None]] = []
    for vid in videos:
        p1 = run_yolo_phase1(vid, yolo_model_path)
        if not p1.complete or p1.T_star is None:
            log.warning("  %s: Phase1 FAIL", vid.name)
            token_list.append((vid.name, None))
            continue
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, p1.T_star)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            token_list.append((vid.name, None))
            continue
        crop, _ = detect_lcd(frame)
        if crop is None:
            token_list.append((vid.name, None))
            continue
        tokens = _extract_patch_tokens(crop, processor, dino_model, device)
        token_list.append((vid.name, tokens))
        log.info("  OK  %s", vid.name)

    valid = [(name, t) for name, t in token_list if t is not None]
    if len(valid) < 2:
        raise RuntimeError("Too few valid calibration videos for LOO")

    loo_scores: list[float] = []
    for i, (held_name, held_tok) in enumerate(valid):
        other = np.concatenate([t for j, (_, t) in enumerate(valid) if j != i], axis=0).astype(np.float32)
        other_norm = other / (np.linalg.norm(other, axis=1, keepdims=True) + 1e-8)
        score = _anomaly_score_from_norm(held_tok, other_norm)
        loo_scores.append(score)
        log.info("  LOO[%d/%d]  %-30s  score=%.4f", i + 1, len(valid), held_name, score)

    thr = float(np.percentile(loo_scores, cal_pct))
    log.info(
        "LOO: min=%.4f  mean=%.4f  p%.0f=%.4f  max=%.4f",
        min(loo_scores), float(np.mean(loo_scores)), cal_pct, thr, max(loo_scores),
    )

    out_data = {
        "threshold":      round(thr, 6),
        "cal_percentile": cal_pct,
        "n_videos":       len(valid),
        "model_id":       model_id,
        "loo_scores":     [round(s, 4) for s in loo_scores],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    log.info("Threshold saved → %s  (thr=%.4f)", out_path, thr)
    return thr


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase2DinoResult:
    passed:        bool
    verdict:       str          # PASS | FAIL
    scores:        list[float]  # per-frame anomaly scores
    median_score:  float
    max_score:     float
    fail_count:    int          # frames above threshold
    dino_thr:      float
    n_timepoints:  int
    frame_indices: list[int] = field(default_factory=list)
    error:         str | None = None


# ---------------------------------------------------------------------------
# Inference entry point
# ---------------------------------------------------------------------------

def run_phase2_dino(
    video_path:   str | Path,
    T_star:       int,
    bank_path:    str | Path,
    thr_path:     str | Path,
    model_id:     str   = DINO_MODEL_ID,
    n_timepoints: int   = N_TIMEPOINTS,
    confirm_hits: int   = CONFIRM_HITS,
    fps:          float = 12.0,
    device:       str   = "cpu",
) -> Phase2DinoResult:
    """Run Phase 2B AnomalyDINO on one video.

    Args:
        video_path   : MP4 video
        T_star       : frame index from Phase 1 YOLO
        bank_path    : .npz reference bank (key "bank") from build_reference_bank()
        thr_path     : JSON with key "threshold" from calibrate_dino_threshold()
        confirm_hits : frames above threshold needed to FAIL (default 4 of 5)
    """
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    try:
        with open(thr_path) as f:
            dino_thr = float(json.load(f)["threshold"])
    except Exception as exc:
        return Phase2DinoResult(
            passed=False, verdict="FAIL",
            scores=[], median_score=0.0, max_score=0.0,
            fail_count=0, dino_thr=0.0, n_timepoints=0,
            error=f"thr_load_error:{exc}",
        )

    try:
        ref_norm = _get_bank_norm(Path(bank_path))
    except Exception as exc:
        return Phase2DinoResult(
            passed=False, verdict="FAIL",
            scores=[], median_score=0.0, max_score=0.0,
            fail_count=0, dino_thr=dino_thr, n_timepoints=0,
            error=f"bank_load_error:{exc}",
        )

    try:
        processor, dino_model = _get_dino(model_id, device)
    except Exception as exc:
        return Phase2DinoResult(
            passed=False, verdict="FAIL",
            scores=[], median_score=0.0, max_score=0.0,
            fail_count=0, dino_thr=dino_thr, n_timepoints=0,
            error=f"dino_load_error:{exc}",
        )

    frame_idxs = list(range(T_star, T_star + n_timepoints))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase2DinoResult(
            passed=False, verdict="FAIL",
            scores=[], median_score=0.0, max_score=0.0,
            fail_count=0, dino_thr=dino_thr, n_timepoints=0,
            error=f"cannot_open:{video_path}",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores:     list[float] = []
    valid_idxs: list[int]   = []

    for fidx in frame_idxs:
        if fidx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        crop, _ = detect_lcd(frame)
        if crop is None:
            log.debug("Phase2Dino: detect_lcd failed at frame %d — skip", fidx)
            continue
        tokens = _extract_patch_tokens(crop, processor, dino_model, device)
        scores.append(_anomaly_score_from_norm(tokens, ref_norm))
        valid_idxs.append(fidx)

    cap.release()

    if not scores:
        return Phase2DinoResult(
            passed=False, verdict="FAIL",
            scores=[], median_score=0.0, max_score=0.0,
            fail_count=0, dino_thr=dino_thr, n_timepoints=0,
            error="no_valid_lcd_crops",
        )

    fail_count   = sum(s > dino_thr for s in scores)
    median_score = float(np.median(scores))
    max_score    = float(max(scores))
    passed       = fail_count < confirm_hits
    verdict      = "PASS" if passed else "FAIL"

    log.info(
        "Phase2Dino: verdict=%s  fail_frames=%d/%d  median=%.4f  max=%.4f  thr=%.4f",
        verdict, fail_count, len(scores), median_score, max_score, dino_thr,
    )
    return Phase2DinoResult(
        passed        = passed,
        verdict       = verdict,
        scores        = [round(s, 4) for s in scores],
        median_score  = round(median_score, 4),
        max_score     = round(max_score, 4),
        fail_count    = fail_count,
        dino_thr      = round(dino_thr, 4),
        n_timepoints  = len(scores),
        frame_indices = valid_idxs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    ap = argparse.ArgumentParser(description="Phase 2B — AnomalyDINO calibration tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build-bank",
                        help="Build DINOv2 reference bank from GOOD calibration videos "
                             "(Jul-14 + Sep-09 ONLY, NEVER Sep-15)")
    pb.add_argument("--dataset",    required=True,
                    help="Comma-separated dirs of GOOD MP4 videos")
    pb.add_argument("--yolo-model", required=True)
    pb.add_argument("--out",        default="models/dino_reference_bank.npz")
    pb.add_argument("--model-id",   default=DINO_MODEL_ID)
    pb.add_argument("--n-samples",  type=int, default=20)
    pb.add_argument("--device",     default="cpu")

    pc = sub.add_parser("calibrate",
                        help="Leave-one-out calibration of anomaly threshold "
                             "(Jul-14 + Sep-09 ONLY, NEVER Sep-15)")
    pc.add_argument("--dataset",    required=True,
                    help="Comma-separated dirs of GOOD MP4 videos")
    pc.add_argument("--yolo-model", required=True)
    pc.add_argument("--out",        default="models/dino_threshold.json")
    pc.add_argument("--model-id",   default=DINO_MODEL_ID)
    pc.add_argument("--n-samples",  type=int,   default=20)
    pc.add_argument("--cal-pct",    type=float, default=CAL_PERCENTILE)
    pc.add_argument("--device",     default="cpu")

    args = ap.parse_args()
    dirs = [Path(d.strip()) for d in args.dataset.split(",") if d.strip()]

    if args.cmd == "build-bank":
        build_reference_bank(
            dataset_dirs    = dirs,
            yolo_model_path = Path(args.yolo_model),
            out_path        = Path(args.out),
            model_id        = args.model_id,
            n_samples       = args.n_samples,
            device          = args.device,
        )
    elif args.cmd == "calibrate":
        calibrate_dino_threshold(
            dataset_dirs    = dirs,
            yolo_model_path = Path(args.yolo_model),
            out_path        = Path(args.out),
            model_id        = args.model_id,
            n_samples       = args.n_samples,
            cal_pct         = args.cal_pct,
            device          = args.device,
        )


if __name__ == "__main__":
    main()
