"""Evaluation script for v6c pipeline on locked test sets (Jul-14, Sep-09, Sep-15).

Calibration commands (run ONCE on Jul-14 + Sep-09 GOOD only — never Sep-15):
    # Per-element thresholds (Part A)
    python -m server_v6c.phase2_elements \\
        --dataset '/path/Jul-14/GOOD,/path/Sep-09/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --out models/element_thresholds.json

    # DINOv2 reference bank (Part B)
    python -m server_v6c.phase2_dino build-bank \\
        --dataset '/path/Jul-14/GOOD,/path/Sep-09/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --out models/dino_reference_bank.npz

    # DINOv2 threshold (LOO calibration, Part B)
    python -m server_v6c.phase2_dino calibrate \\
        --dataset '/path/Jul-14/GOOD,/path/Sep-09/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --out models/dino_threshold.json

    # Rebuild golden template (for legacy SSIM/IoU paths, Jul-14+Sep-09 only)
    python -m server_v6c.build_golden_template \\
        --dataset '/path/Jul-14/GOOD,/path/Sep-09/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --out models/golden_template.npz --domain Jul14+Sep09

Evaluation (Sep-15 — GOOD):
    python -m server_v6c.eval \\
        --dataset '/home/om/src/FDU Dataset/Sep-15-2026/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --phase3-model models/v6a/best.pth \\
        --phase2-elements --element-thresholds models/element_thresholds.json \\
        --phase2-dino --dino-bank models/dino_reference_bank.npz \\
                      --dino-thr  models/dino_threshold.json \\
        --all-good --name Sep15-GOOD-final --fps 12

Evaluation (Sep-15 — NOT GOOD):
    python -m server_v6c.eval \\
        --dataset '/home/om/src/FDU Dataset/Sep-15-2026/NOT GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --phase3-model models/v6a/best.pth \\
        --phase2-elements --element-thresholds models/element_thresholds.json \\
        --phase2-dino --dino-bank models/dino_reference_bank.npz \\
                      --dino-thr  models/dino_threshold.json \\
        --all-not-good --name Sep15-NOTGOOD-final --fps 12

IMPORTANT: Jul-14 and Sep-09 are locked eval sets. Sep-15 is also locked.
           NEVER use any of them for training or calibration.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Built-in Jul-14 ground truth (12-FPS timestamps)
_JUL14_LABELS: dict[str, str] = {
    "141138": "GOOD",     "141341": "NOT_GOOD", "141950": "NOT_GOOD",
    "142150": "GOOD",     "142516": "NOT_GOOD", "142739": "GOOD",
    "143041": "GOOD",     "143208": "GOOD",     "143426": "NOT_GOOD",
    "143604": "NOT_GOOD",
}

# Sep-09 ground truth (all GOOD)
_SEP09_LABELS: dict[str, str] = {
    "200811": "GOOD", "201244": "GOOD", "201351": "GOOD", "201454": "GOOD",
    "201608": "GOOD", "201704": "GOOD", "201759": "GOOD", "201854": "GOOD",
    "201950": "GOOD", "202046": "GOOD", "202145": "GOOD", "202244": "GOOD",
    "202345": "GOOD",
}


def _find_label(stem: str, label_map: dict[str, str]) -> str | None:
    if stem in label_map:
        return label_map[stem]
    for fragment, lbl in label_map.items():
        if fragment in stem:
            return lbl
    return None


def _get_labels(name: str, all_good: bool, labels_path: str | None, all_not_good: bool = False) -> dict[str, str]:
    if all_good:
        return {"__all_good__": "GOOD"}
    if all_not_good:
        return {"__all_not_good__": "NOT_GOOD"}
    if labels_path:
        with open(labels_path) as f:
            raw = json.load(f)
        return {Path(k).stem if k.endswith(".mp4") else k: v for k, v in raw.items()}
    nl = name.lower()
    if "jul" in nl or "14" in nl:
        log.info("Using built-in Jul-14 labels")
        return _JUL14_LABELS
    if "sep" in nl or "09" in nl:
        log.info("Using built-in Sep-09 labels (all GOOD)")
        return _SEP09_LABELS
    raise ValueError("Provide --labels, --all-good, --all-not-good, or use --name Jul-14 / Sep-09")


def main() -> None:
    ap = argparse.ArgumentParser(description="v6c pipeline evaluation")
    ap.add_argument("--dataset",       required=True,  help="Folder of MP4 videos")
    ap.add_argument("--yolo-model",    required=True,  help="Path to YOLO best.pt")
    ap.add_argument("--phase3-model",  required=True,  help="Path to models/v6a/best.pth")

    # ── New primary Phase 2: Elements (2A) + AnomalyDINO (2B) ───────────────
    ap.add_argument("--phase2-elements",    action="store_true",
                    help="Enable Phase 2A per-element presence check")
    ap.add_argument("--element-thresholds", default=None,
                    help="Path to element_thresholds.json (calibrated on Jul-14+Sep-09 GOOD)")
    ap.add_argument("--elements-weak-k",    type=int,   default=3,
                    help="Fail if >= this many AMBIGUOUS elements (default 3)")
    ap.add_argument("--elements-contrast-thr", type=int, default=15,
                    help="Intensity delta vs collar to count as lit (default 15)")
    ap.add_argument("--phase2-dino",        action="store_true",
                    help="Enable Phase 2B AnomalyDINO whole-crop check")
    ap.add_argument("--dino-bank",          default=None,
                    help="Path to dino_reference_bank.npz (built from Jul-14+Sep-09 GOOD)")
    ap.add_argument("--dino-thr",           default=None,
                    help="Path to dino_threshold.json (LOO calibrated on Jul-14+Sep-09 GOOD)")
    ap.add_argument("--dino-model-id",      default="facebook/dinov2-small")
    ap.add_argument("--dino-confirm-hits",  type=int, default=4,
                    help="Frames above threshold needed to FAIL (default 4/5)")

    # ── Legacy Phase 2 paths ─────────────────────────────────────────────────
    ap.add_argument("--phase2-ssim",    action="store_true",
                    help="Enable Phase 2 SSIM check (requires --phase2-template)")
    ap.add_argument("--phase2-ssim-ghost-thr",   type=float, default=0.40)
    ap.add_argument("--phase2-ssim-thr-margin",  type=float, default=0.97)
    ap.add_argument("--phase2-template", default=None,
                    help="Path to golden_template.npz (SSIM or IoU)")
    ap.add_argument("--phase2-template-iou-thr",   type=float, default=0.20)
    ap.add_argument("--phase2-template-ghost-thr",  type=float, default=0.40)
    ap.add_argument("--phase2-template-roi-boost", action="store_true")
    ap.add_argument("--phase2-roi",    action="store_true",
                    help="Enable Phase 2 ROI coverage check")
    ap.add_argument("--phase2-roi-ghost-thr", type=float, default=0.40)
    ap.add_argument("--phase2-model",  default=None,
                    help="Legacy Phase 2: path to dts_p2v2_best.pth")

    # ── Common ───────────────────────────────────────────────────────────────
    ap.add_argument("--name",          default="eval", help="Run name")
    ap.add_argument("--labels",        default=None,   help="JSON {stem: GOOD|NOT_GOOD}")
    ap.add_argument("--all-good",      action="store_true")
    ap.add_argument("--all-not-good",  action="store_true")
    ap.add_argument("--yolo-conf",     type=float, default=0.50)
    ap.add_argument("--phase3-thr",    type=float, default=0.50)
    ap.add_argument("--phase2-thr",    type=float, default=0.50)
    ap.add_argument("--n-phase3-frames", type=int, default=3)
    ap.add_argument("--fps",           type=float, default=12.0)
    ap.add_argument("--out-dir",       default="data/v6c_results")
    args = ap.parse_args()

    from .pipeline import V6cPipeline
    from .phase2_template import RECOMMENDED_PER_ROI_THR

    per_roi = RECOMMENDED_PER_ROI_THR if args.phase2_template_roi_boost else None

    pipe = V6cPipeline(
        yolo_model_path                  = args.yolo_model,
        phase3_model_path                = args.phase3_model,
        use_phase2_elements              = args.phase2_elements,
        elements_threshold_path          = args.element_thresholds,
        elements_weak_k                  = args.elements_weak_k,
        elements_contrast_thr            = args.elements_contrast_thr,
        use_phase2_dino                  = args.phase2_dino,
        dino_bank_path                   = args.dino_bank,
        dino_thr_path                    = args.dino_thr,
        dino_model_id                    = args.dino_model_id,
        dino_confirm_hits                = args.dino_confirm_hits,
        use_phase2_ssim                  = args.phase2_ssim,
        phase2_ssim_ghost_thr            = args.phase2_ssim_ghost_thr,
        phase2_ssim_thr_margin           = args.phase2_ssim_thr_margin,
        phase2_template_path             = args.phase2_template,
        phase2_template_iou_thr          = args.phase2_template_iou_thr,
        phase2_template_ghost_thr        = args.phase2_template_ghost_thr,
        phase2_template_per_roi_iou_thr  = per_roi,
        use_phase2_roi                   = args.phase2_roi,
        phase2_roi_ghost_thr             = args.phase2_roi_ghost_thr,
        phase2_model_path                = args.phase2_model,
        yolo_conf                        = args.yolo_conf,
        phase2_thr                       = args.phase2_thr,
        phase3_thr                       = args.phase3_thr,
        fps                              = args.fps,
        n_phase3_frames                  = args.n_phase3_frames,
    )

    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _get_labels(args.name, args.all_good, args.labels, args.all_not_good)
    videos = sorted(p for p in dataset.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        raise RuntimeError(f"No video files found in {dataset}")

    results: list[dict] = []
    correct = 0
    labeled = 0

    sep = "─" * 88
    # Build column header based on enabled checks
    p2elem_col = "  P2ELEM" if args.phase2_elements   else ""
    p2dino_col = "  P2DINO" if args.phase2_dino        else ""
    p2ssim_col = "  P2SSIM" if args.phase2_ssim        else ""
    p2tmpl_col = "  P2TMPL" if (args.phase2_template and not args.phase2_ssim) else ""
    p2roi_col  = "  P2ROI"  if args.phase2_roi         else ""
    p3_col     = "" if (args.phase2_elements or args.phase2_dino) else " {'P3_pf':>7}"
    hdr = (
        f"{'Video':<22} {'GT':>8} {'Verdict':>8} {'P1':>4}"
        f"{p2elem_col}{p2dino_col}{p2ssim_col}{p2tmpl_col}{p2roi_col}"
        f" {'P3_pf':>7} {'ms':>6} {'OK':>3}"
    )

    print(f"\n{sep}")
    print(f"  v6c  —  {args.name}")
    print(f"  YOLO: {Path(args.yolo_model).name}   Phase3: {Path(args.phase3_model).name}")
    if args.phase2_elements:
        thr_name = Path(args.element_thresholds).name if args.element_thresholds else "MISSING"
        print(f"  Phase2A Elements: {thr_name}  weak_k={args.elements_weak_k}")
    if args.phase2_dino:
        bank_name = Path(args.dino_bank).name if args.dino_bank else "MISSING"
        thr_name  = Path(args.dino_thr).name  if args.dino_thr  else "MISSING"
        print(f"  Phase2B DINO: bank={bank_name}  thr={thr_name}  confirm_hits={args.dino_confirm_hits}")
    if args.phase2_ssim and args.phase2_template:
        print(f"  Phase2 SSIM: {Path(args.phase2_template).name}  ghost_thr={args.phase2_ssim_ghost_thr}  margin={args.phase2_ssim_thr_margin}")
    elif args.phase2_template:
        print(f"  Phase2 Template (IoU): {Path(args.phase2_template).name}  iou_thr={args.phase2_template_iou_thr}")
    if args.phase2_roi:
        print(f"  Phase2 ROI: enabled  ghost_thr={args.phase2_roi_ghost_thr}")
    if args.phase2_model:
        print(f"  Phase2 legacy: {Path(args.phase2_model).name}")
    print(sep)
    print(hdr)
    print(sep)

    for vid in videos:
        if labels.get("__all_good__"):
            gt = "GOOD"
        elif labels.get("__all_not_good__"):
            gt = "NOT_GOOD"
        else:
            gt = _find_label(vid.stem, labels)

        result  = pipe.predict(vid)
        verdict = result["verdict"]
        p1      = result["phase1"]
        p2_elem = result.get("phase2_elements")
        p2_dino = result.get("phase2_dino")
        p2_ssim = result.get("phase2_ssim")
        p2_tmpl = result.get("phase2_template")
        p2_roi  = result.get("phase2_roi")
        p3      = result["phase3"]

        p1_ok  = "✓" if p1 and p1.get("complete") else "✗"
        p3_pf  = f"{p3['median_prob_fail']:.3f}" if p3 else "  N/A"
        ms_s   = f"{result['inference_ms']:.0f}"

        p2elem_s = ""
        if args.phase2_elements:
            if p2_elem:
                p2elem_s = f"  {p2_elem['verdict'][:4]:>6}"
            else:
                p2elem_s = f"  {'N/A':>6}"

        p2dino_s = ""
        if args.phase2_dino:
            if p2_dino:
                score_s = f"{p2_dino['max_score']:.3f}"
                tag     = "FAIL" if not p2_dino["passed"] else "PASS"
                p2dino_s = f"  {tag[:4]:>4}({score_s})"
            else:
                p2dino_s = f"  {'N/A':>9}"

        p2ssim_s = ""
        if args.phase2_ssim:
            p2ssim_s = f"  {p2_ssim['verdict'][:4]:>6}" if p2_ssim else f"  {'N/A':>6}"

        p2tmpl_s = ""
        if args.phase2_template and not args.phase2_ssim:
            p2tmpl_s = f"  {p2_tmpl['verdict'][:4]:>6}" if p2_tmpl else f"  {'N/A':>6}"

        p2roi_s = ""
        if args.phase2_roi:
            p2roi_s = f"  {p2_roi['verdict'][:4]:>6}" if p2_roi else f"  {'N/A':>6}"

        is_correct = None
        if gt is not None:
            labeled += 1
            expected_pass = (gt == "GOOD")
            if result["passed"] == expected_pass:
                is_correct = True
                correct += 1
            else:
                is_correct = False

        ok_sym = ("✓" if is_correct else "✗") if is_correct is not None else "?"
        gt_s   = gt or "?"

        print(
            f"  {vid.stem:<22} {gt_s:>8} {verdict:>8} {p1_ok:>4}"
            f"{p2elem_s}{p2dino_s}{p2ssim_s}{p2tmpl_s}{p2roi_s}"
            f" {p3_pf:>7} {ms_s:>6} {ok_sym:>3}"
        )

        # Phase 1 failure detail
        if p1 and not p1.get("complete"):
            missing = [c for c, v in p1.get("confirmed", {}).items() if not v]
            confs   = {c: f"{p1['checklist'].get(c, 0):.2f}" for c in missing}
            print(f"    [Phase1 FAIL] missing={missing}  best_conf={confs}")

        # Phase 2A elements failure detail
        if p2_elem and not p2_elem["passed"]:
            miss = p2_elem.get("confident_miss", [])
            weak = p2_elem.get("weak_elements", [])
            print(f"    [P2ELEM FAIL] confident_miss={miss}  weak({p2_elem['n_weak']})={weak[:5]}")

        # Phase 2B DINO failure detail
        if p2_dino and not p2_dino["passed"]:
            print(
                f"    [P2DINO FAIL] fail_frames={p2_dino['fail_count']}/{p2_dino['n_timepoints']}"
                f"  scores={p2_dino['scores']}  thr={p2_dino['dino_thr']:.4f}"
            )

        # Legacy SSIM failure / ambiguous
        if p2_ssim and result.get("failed_at") == "phase2_ssim":
            print(f"    [P2SSIM FAIL] fail_rois={p2_ssim.get('confirmed_fail_rois', [])}  "
                  f"ghost={p2_ssim.get('ghost_frac', 0):.4f}  "
                  f"ambiguous={p2_ssim.get('ambiguous_rois', [])}")
        if p2_ssim and p2_ssim.get("verdict") == "AMBIGUOUS":
            print(f"    [P2SSIM AMBIGUOUS→P3] ambiguous={p2_ssim.get('ambiguous_rois', [])}  "
                  f"on={p2_ssim.get('confirmed_on_count', 0)}/{len(p2_ssim.get('roi_states', []))}")

        # Legacy template failure
        if p2_tmpl and result.get("failed_at") == "phase2_tmpl":
            print(f"    [P2TMPL FAIL] fail_rois={p2_tmpl.get('confirmed_fail_rois', [])}  "
                  f"ghost={p2_tmpl.get('ghost_frac', 0):.4f}")

        # Legacy ROI failure / ambiguous
        if p2_roi and result.get("failed_at") == "phase2_roi":
            print(f"    [P2ROI FAIL] fail_rois={p2_roi.get('confirmed_fail_rois', [])}  "
                  f"ghost={p2_roi.get('ghost_frac', 0):.4f}")
        if p2_roi and p2_roi.get("verdict") == "AMBIGUOUS":
            print(f"    [P2ROI AMBIGUOUS→P3] ambiguous={p2_roi.get('ambiguous_rois', [])}  "
                  f"on={p2_roi.get('confirmed_on_count', 0)}/{len(p2_roi.get('roi_states', []))}")

        # Phase 3 failure detail
        if p3 and verdict == "FAIL" and result.get("failed_at") == "phase3":
            print(f"    [Phase3 FAIL] per_frame={p3['per_frame_probs']}  median={p3['median_prob_fail']:.4f}")

        results.append({
            "video":   vid.name,
            "stem":    vid.stem,
            "gt":      gt,
            "verdict": verdict,
            "passed":  result["passed"],
            "correct": is_correct,
            "failed_at": result.get("failed_at"),
            # Phase 2A elements
            "phase2_elements_verdict":   p2_elem.get("verdict") if p2_elem else None,
            "phase2_elements_miss":      p2_elem.get("confident_miss") if p2_elem else None,
            "phase2_elements_weak":      p2_elem.get("weak_elements") if p2_elem else None,
            "phase2_elements_n_weak":    p2_elem.get("n_weak") if p2_elem else None,
            # Phase 2B DINO
            "phase2_dino_verdict":       p2_dino.get("verdict") if p2_dino else None,
            "phase2_dino_max_score":     p2_dino.get("max_score") if p2_dino else None,
            "phase2_dino_median_score":  p2_dino.get("median_score") if p2_dino else None,
            "phase2_dino_fail_count":    p2_dino.get("fail_count") if p2_dino else None,
            "phase2_dino_scores":        p2_dino.get("scores") if p2_dino else None,
            # Legacy SSIM
            "phase2_ssim_verdict":       p2_ssim.get("verdict") if p2_ssim else None,
            "phase2_ssim_confirmed_on":  p2_ssim.get("confirmed_on_count") if p2_ssim else None,
            "phase2_ssim_fail_rois":     p2_ssim.get("confirmed_fail_rois") if p2_ssim else None,
            "phase2_ssim_ambiguous":     p2_ssim.get("ambiguous_rois") if p2_ssim else None,
            "phase2_ssim_ghost_frac":    p2_ssim.get("ghost_frac") if p2_ssim else None,
            # Legacy ROI
            "phase2_roi_verdict":        p2_roi.get("verdict") if p2_roi else None,
            "phase2_roi_confirmed_on":   p2_roi.get("confirmed_on_count") if p2_roi else None,
            "phase2_roi_fail_rois":      p2_roi.get("confirmed_fail_rois") if p2_roi else None,
            "phase2_roi_ambiguous":      p2_roi.get("ambiguous_rois") if p2_roi else None,
            "phase2_roi_ghost_frac":     p2_roi.get("ghost_frac") if p2_roi else None,
            # Phase 3
            "phase3_median_prob_fail":   p3.get("median_prob_fail") if p3 else None,
            "phase3_per_frame":          p3.get("per_frame_probs") if p3 else None,
            # Timing
            "inference_ms":              result["inference_ms"],
        })

    print(sep)
    acc_s = f"{correct}/{labeled}" if labeled else "N/A"
    pct_s = f" ({100*correct/labeled:.0f}%)" if labeled else ""
    print(f"  Result: {acc_s}{pct_s}")
    print(f"{sep}\n")

    summary = {
        "name":                  args.name,
        "yolo_model":            str(args.yolo_model),
        "phase3_model":          str(args.phase3_model),
        "phase2_elements":       args.phase2_elements,
        "element_thresholds":    str(args.element_thresholds) if args.element_thresholds else None,
        "elements_weak_k":       args.elements_weak_k,
        "phase2_dino":           args.phase2_dino,
        "dino_bank":             str(args.dino_bank) if args.dino_bank else None,
        "dino_thr":              str(args.dino_thr) if args.dino_thr else None,
        "dino_confirm_hits":     args.dino_confirm_hits,
        "phase2_ssim":           args.phase2_ssim,
        "phase2_template":       str(args.phase2_template) if args.phase2_template else None,
        "phase2_roi":            args.phase2_roi,
        "phase2_model":          str(args.phase2_model) if args.phase2_model else None,
        "total":                 len(results),
        "labeled":               labeled,
        "correct":               correct,
        "accuracy":              round(correct / labeled, 4) if labeled else None,
        "videos":                results,
    }
    out_file = out_dir / f"eval_{args.name.replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Report saved → %s", out_file)


if __name__ == "__main__":
    main()
