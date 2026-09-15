"""Evaluation script for v6c pipeline on locked test sets (Jul-14, Sep-09).

Usage — Jul-14:
    python -m server_v6c.eval \\
        --dataset      /path/to/July-14-2026/12-FPS \\
        --yolo-model   data/macro_dataset/runs/macro_test/weights/best.pt \\
        --phase3-model models/v6a/best.pth \\
        --name         Jul-14

Usage — Sep-09 (all GOOD):
    python -m server_v6c.eval \\
        --dataset      /path/to/Sep-09-2026 \\
        --yolo-model   data/macro_dataset/runs/macro_test/weights/best.pt \\
        --phase3-model models/v6a/best.pth \\
        --all-good     --name Sep-09

Enable Phase 2 ROI check (new — 8-region coverage + ghost):
    Add --phase2-roi

Enable legacy Phase 2 (masked EfficientNet):
    Add --phase2-model models/dts_p2v2_best.pth

IMPORTANT: Jul-14 and Sep-09 are locked eval sets. NEVER use them for training.
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


def _get_labels(name: str, all_good: bool, labels_path: str | None) -> dict[str, str]:
    if all_good:
        return {"__all_good__": "GOOD"}
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
    raise ValueError("Provide --labels, --all-good, or use --name Jul-14 / Sep-09")


def main() -> None:
    ap = argparse.ArgumentParser(description="v6c pipeline evaluation")
    ap.add_argument("--dataset",       required=True,  help="Folder of MP4 videos")
    ap.add_argument("--yolo-model",    required=True,  help="Path to YOLO best.pt")
    ap.add_argument("--phase3-model",  required=True,  help="Path to models/v6a/best.pth")
    ap.add_argument("--phase2-template", default=None,
                    help="Fix 2: path to golden_template.npz (IoU content check)")
    ap.add_argument("--phase2-template-iou-thr",   type=float, default=0.20,
                    help="IoU threshold for template match (default 0.20)")
    ap.add_argument("--phase2-template-ghost-thr",  type=float, default=0.40,
                    help="Ghost pixel threshold for template check (default 0.40)")
    ap.add_argument("--phase2-template-roi-boost", action="store_true",
                    help="Enable per-ROI overrides: left_clock=0.35, center_88=0.30 "
                         "(other ROIs use --phase2-template-iou-thr)")
    ap.add_argument("--phase2-roi",    action="store_true", help="Enable Phase 2 ROI coverage check")
    ap.add_argument("--phase2-roi-ghost-thr", type=float, default=0.40,
                    help="Ghost pixel fraction threshold for Phase 2 ROI (default 0.40)")
    ap.add_argument("--phase2-model",  default=None,   help="Legacy Phase 2: path to dts_p2v2_best.pth")
    ap.add_argument("--name",          default="eval", help="Run name (Jul-14 / Sep-09)")
    ap.add_argument("--labels",        default=None,   help="JSON {stem: GOOD|NOT_GOOD}")
    ap.add_argument("--all-good",      action="store_true")
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

    dataset   = Path(args.dataset)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels    = _get_labels(args.name, args.all_good, args.labels)
    videos    = sorted(p for p in dataset.iterdir() if p.suffix.lower() in _VIDEO_EXTS)

    if not videos:
        raise RuntimeError(f"No video files found in {dataset}")

    results: list[dict] = []
    correct = 0
    labeled = 0

    sep = "─" * 80
    p2tmpl_col = "  P2TMPL" if args.phase2_template else ""
    p2roi_col  = "  P2ROI"  if args.phase2_roi      else ""
    hdr = f"{'Video':<22} {'GT':>8} {'Verdict':>8} {'P1':>4}{p2tmpl_col}{p2roi_col} {'P3_pf':>7} {'ms':>6} {'OK':>3}"
    print(f"\n{sep}")
    print(f"  v6c  —  {args.name}")
    print(f"  YOLO: {Path(args.yolo_model).name}   Phase3: {Path(args.phase3_model).name}")
    if args.phase2_template:
        roi_boost_s = f"  roi_boost={per_roi}" if per_roi else ""
        print(f"  Phase2 Template: {Path(args.phase2_template).name}  iou_thr={args.phase2_template_iou_thr}{roi_boost_s}")
    if args.phase2_roi:
        print(f"  Phase2 ROI: enabled  ghost_thr={args.phase2_roi_ghost_thr}")
    if args.phase2_model:
        print(f"  Phase2 legacy: {Path(args.phase2_model).name}")
    print(sep)
    print(hdr)
    print(sep)

    for vid in videos:
        gt = "GOOD" if labels.get("__all_good__") else _find_label(vid.stem, labels)

        result  = pipe.predict(vid)
        verdict = result["verdict"]
        p1      = result["phase1"]
        p2_tmpl = result.get("phase2_template")
        p2_roi  = result.get("phase2_roi")
        p3      = result["phase3"]

        p1_ok    = "✓" if p1 and p1.get("complete") else "✗"
        p3_pf    = f"{p3['median_prob_fail']:.3f}" if p3 else "  N/A"
        ms_s     = f"{result['inference_ms']:.0f}"

        p2tmpl_s = ""
        if args.phase2_template:
            if p2_tmpl:
                p2tmpl_s = f"  {p2_tmpl['verdict'][:4]:>6}"
            else:
                p2tmpl_s = f"  {'N/A':>6}"

        p2roi_s = ""
        if args.phase2_roi:
            if p2_roi:
                p2roi_s = f"  {p2_roi['verdict'][:4]:>6}"
            else:
                p2roi_s = f"  {'N/A':>6}"

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

        print(f"  {vid.stem:<22} {gt_s:>8} {verdict:>8} {p1_ok:>4}{p2tmpl_s}{p2roi_s} {p3_pf:>7} {ms_s:>6} {ok_sym:>3}")

        # Show Phase 1 failures
        if p1 and not p1.get("complete"):
            missing = [c for c, v in p1.get("confirmed", {}).items() if not v]
            confs   = {c: f"{p1['checklist'].get(c, 0):.2f}" for c in missing}
            print(f"    [Phase1 FAIL] missing: {missing} | best_conf: {confs}")

        # Show Phase 2 template failures
        if p2_tmpl and result.get("failed_at") == "phase2_tmpl":
            fail_rois = p2_tmpl.get("confirmed_fail_rois", [])
            ghost     = p2_tmpl.get("ghost_frac", 0)
            print(f"    [P2TMPL FAIL] fail_rois={fail_rois}  ghost={ghost:.4f}  "
                  f"ambiguous={p2_tmpl.get('ambiguous_rois', [])}")

        # Show Phase 2 ROI failures
        if p2_roi and result.get("failed_at") == "phase2_roi":
            fail_rois = p2_roi.get("confirmed_fail_rois", [])
            ghost     = p2_roi.get("ghost_frac", 0)
            print(f"    [P2ROI FAIL] fail_rois={fail_rois}  ghost={ghost:.4f}  "
                  f"ambiguous={p2_roi.get('ambiguous_rois', [])}")

        # Show Phase 2 ROI detail on AMBIGUOUS (routed to Phase 3)
        if p2_roi and p2_roi.get("verdict") == "AMBIGUOUS":
            print(f"    [P2ROI AMBIGUOUS→P3] ambiguous={p2_roi.get('ambiguous_rois', [])}  "
                  f"on={p2_roi.get('confirmed_on_count', 0)}/{len(p2_roi.get('roi_states', []))}")

        # Show Phase 3 per-frame breakdown on FAIL
        if p3 and verdict == "FAIL" and result.get("failed_at") == "phase3":
            print(f"    [Phase3 FAIL] per_frame: {p3['per_frame_probs']}  median={p3['median_prob_fail']:.4f}")

        results.append({
            "video":   vid.name,
            "stem":    vid.stem,
            "gt":      gt,
            "verdict": verdict,
            "passed":  result["passed"],
            "correct": is_correct,
            "phase1_complete":   p1.get("complete") if p1 else None,
            "phase1_checklist":  p1.get("checklist") if p1 else None,
            "phase2_roi_verdict":       p2_roi.get("verdict") if p2_roi else None,
            "phase2_roi_confirmed_on":  p2_roi.get("confirmed_on_count") if p2_roi else None,
            "phase2_roi_fail_rois":     p2_roi.get("confirmed_fail_rois") if p2_roi else None,
            "phase2_roi_ambiguous":     p2_roi.get("ambiguous_rois") if p2_roi else None,
            "phase2_roi_ghost_frac":    p2_roi.get("ghost_frac") if p2_roi else None,
            "phase3_median_prob_fail":  p3.get("median_prob_fail") if p3 else None,
            "phase3_per_frame":         p3.get("per_frame_probs") if p3 else None,
            "failed_at":    result.get("failed_at"),
            "inference_ms": result["inference_ms"],
        })

    print(sep)
    acc_s = f"{correct}/{labeled}" if labeled else "N/A"
    pct_s = f" ({100*correct/labeled:.0f}%)" if labeled else ""
    print(f"  Result: {acc_s}{pct_s}")
    print(f"{sep}\n")

    # Save report
    summary = {
        "name":              args.name,
        "yolo_model":        str(args.yolo_model),
        "phase3_model":      str(args.phase3_model),
        "phase2_roi":        args.phase2_roi,
        "phase2_roi_ghost_thr": args.phase2_roi_ghost_thr if args.phase2_roi else None,
        "phase2_model":      str(args.phase2_model),
        "total":             len(results),
        "labeled":           labeled,
        "correct":           correct,
        "accuracy":          round(correct / labeled, 4) if labeled else None,
        "videos":            results,
    }
    out_file = out_dir / f"eval_{args.name.replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Report saved → %s", out_file)


if __name__ == "__main__":
    main()
