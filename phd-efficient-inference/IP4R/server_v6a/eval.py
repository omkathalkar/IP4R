"""Evaluation script for v6b Method 1 (pure CV) on locked test sets.

Usage — Jul-14 (mixed labels, built-in):
    python -m server_v6a.eval \\
        --dataset    /path/to/July-14-2026/12-FPS \\
        --thresholds data/v6a_results/thresholds.json \\
        --golden     data/reference/lcd_all_on.jpg \\
        --name       Jul-14

Usage — Sep-09 (all GOOD):
    python -m server_v6a.eval \\
        --dataset    /path/to/Sep-09-2026 \\
        --thresholds data/v6a_results/thresholds.json \\
        --golden     data/reference/lcd_all_on.jpg \\
        --all-good   --name Sep-09

labels.json format:  { "141008.mp4": "GOOD", "141138": "NOT_GOOD", ... }

Outputs:
  - Per-video summary table (stdout)
  - Per-element diagnostic rows for FAIL / ABSTAIN (stdout)
  - eval_report.json written to --out-dir
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2

from .atlas import load_atlas
from .pipeline import run_video
from .threshold import load_thresholds

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

_JUL14_LABELS: dict[str, str] = {
    # 30-FPS timestamps
    "141008": "GOOD",      "141414": "NOT_GOOD",  "141917": "NOT_GOOD",
    "142307": "GOOD",      "142443": "NOT_GOOD",  "142812": "GOOD",
    "143007": "GOOD",      "143239": "GOOD",      "143357": "NOT_GOOD",
    "143635": "NOT_GOOD",
    # 12-FPS timestamps (inferred by time-proximity to 30-FPS set)
    "141138": "GOOD",      "141341": "NOT_GOOD",  "141950": "NOT_GOOD",
    "142150": "GOOD",      "142516": "NOT_GOOD",  "142739": "GOOD",
    "143041": "GOOD",      "143208": "GOOD",      "143426": "NOT_GOOD",
    "143604": "NOT_GOOD",
}


def _find_label(stem: str, label_map: dict[str, str]) -> str | None:
    if stem in label_map:
        return label_map[stem]
    for fragment, lbl in label_map.items():
        if fragment in stem:
            return lbl
    return None


def _load_labels(
    labels_path: str | None,
    all_good: bool,
    name: str,
) -> dict[str, str]:
    if all_good:
        return {"__all_good__": "GOOD"}
    if labels_path:
        with open(labels_path) as f:
            raw = json.load(f)
        return {Path(k).stem if k.endswith(".mp4") else k: v for k, v in raw.items()}
    if "jul" in name.lower() or "14" in name:
        log.info("Using built-in Jul-14 labels")
        return _JUL14_LABELS
    raise ValueError(
        "Provide --labels <file>, --all-good, or use --name Jul-14 for built-in labels."
    )


def evaluate(args: argparse.Namespace) -> dict:
    # ── Load assets once ─────────────────────────────────────────────────────
    with open(args.thresholds) as f:
        thr_meta = json.load(f)

    golden_path = args.golden or thr_meta.get("golden_path", "data/reference/lcd_all_on.jpg")
    atlas_path  = args.atlas  or thr_meta.get("atlas_path",  "data/reference/rois.yaml")

    golden_bgr = cv2.imread(str(golden_path))
    if golden_bgr is None:
        raise FileNotFoundError(f"Cannot load golden: {golden_path}")

    atlas      = load_atlas(atlas_path)
    thresholds = load_thresholds(args.thresholds)

    # ── Setup output ─────────────────────────────────────────────────────────
    dataset  = Path(args.dataset)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels  = _load_labels(args.labels, args.all_good, args.name)
    videos  = sorted(p for p in dataset.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        raise RuntimeError(f"No video files in {dataset}")

    results: list[dict] = []
    correct = 0
    total_labeled = 0

    hdr = f"{'Video':<20} {'GT':>8} {'Verdict':>8} {'rho':>5} {'t_sec':>6} {'cov':>5} {'nfail':>5} {'OK':>3}"
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {args.name} — v6b Method 1 (pure CV)")
    print(sep)
    print(hdr)
    print(sep)

    for vid in videos:
        if labels.get("__all_good__"):
            gt = "GOOD"
        else:
            gt = _find_label(vid.stem, labels)

        result = run_video(
            vid, golden_bgr, atlas, thresholds,
            n_candidates=args.n_candidates,
        )

        verdict = result["verdict"]
        rho     = result["ecc_rho"]
        fi      = result.get("frame_info", {})
        t_sec   = fi.get("t_sec", -1)
        cov     = fi.get("coverage", -1)
        n_fail  = len(result.get("failures", []))
        err     = result.get("error")

        is_correct = None
        if gt is not None:
            total_labeled += 1
            gt_pass = (gt == "GOOD")
            if result["passed"] == gt_pass:
                is_correct = True
                correct += 1
            else:
                is_correct = False

        ok_sym  = ("✓" if is_correct else "✗") if is_correct is not None else "?"
        rho_s   = f"{rho:.3f}" if rho > -1 else "  N/A"
        t_s     = f"{t_sec:.1f}" if isinstance(t_sec, float) else str(t_sec)
        cov_s   = f"{cov:.3f}" if isinstance(cov, float) else str(cov)
        gt_s    = gt if gt else "?"

        print(f"  {vid.stem:<20} {gt_s:>8} {verdict:>8} {rho_s:>5} {t_s:>6} {cov_s:>5} {n_fail:>5} {ok_sym:>3}")

        if err:
            print(f"    [!] {err}")

        # Per-element diagnostics for FAIL or ABSTAIN
        if verdict != "PASS" and result["per_element"]:
            failures = result.get("failures", [])
            pe = result["per_element"]
            # Print elements that failed (sorted by nc ascending for quick triage)
            failed_items = sorted(
                [(n, v) for n, v in pe.items() if not v["passed"]],
                key=lambda x: x[1]["nc"],
            )
            for ename, ev in failed_items:
                print(f"      FAIL  {ename:<26}  nc={ev['nc']:.4f}  thr={ev['threshold']:.4f}")

        results.append({
            "video":       vid.name,
            "stem":        vid.stem,
            "gt":          gt,
            "verdict":     verdict,
            "passed":      result["passed"],
            "ecc_rho":     rho,
            "correct":     is_correct,
            "frame_info":  fi,
            "per_element": result.get("per_element", {}),
            "failures":    result.get("failures", []),
            "error":       err,
        })

    print(sep)
    acc_str = f"{correct}/{total_labeled}" if total_labeled else "N/A"
    pct_str = f" ({100*correct/total_labeled:.0f}%)" if total_labeled else ""
    print(f"  Result: {acc_str}{pct_str}")
    print(f"{sep}\n")

    summary = {
        "name":       args.name,
        "thresholds": str(args.thresholds),
        "golden":     str(golden_path),
        "total":      len(results),
        "labeled":    total_labeled,
        "correct":    correct,
        "accuracy":   round(correct / total_labeled, 4) if total_labeled else None,
        "videos":     results,
    }
    out_file = out_dir / f"eval_{args.name.replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Report saved to %s", out_file)

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="IP4R v6b — pure-CV evaluation")
    p.add_argument("--dataset",      required=True, help="Folder of MP4 videos")
    p.add_argument("--thresholds",   required=True, help="thresholds.json from threshold.py")
    p.add_argument("--golden",       default=None,  help="Override golden JPEG path")
    p.add_argument("--atlas",        default=None,  help="Override ROI atlas YAML path")
    p.add_argument("--labels",       default=None,  help="JSON {stem: GOOD|NOT_GOOD}")
    p.add_argument("--all-good",     action="store_true", help="All videos are GOOD")
    p.add_argument("--name",         default="eval", help="Label for this run")
    p.add_argument("--n-candidates", type=int, default=3)
    p.add_argument("--out-dir",      default="data/v6a_results")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
