#!/usr/bin/env python3
"""
nav_stack/eval/run_eval_suite.py — C3 paper comparison table (Phase 8).

Produces a 3-column (+ optional TIC-VLA) table from Phase 7 SLURM results:

  | Method        | Heading Err (°) | ADE (m)     | FDE (m)     | VLA Rate | Latency (s) |
  |---------------|-----------------|-------------|-------------|----------|-------------|
  | Hybrid (ours) |  X.X ± Y.Y      | X.X ± Y.Y   | X.X ± Y.Y   | Z.Z%     | X.X ± Y.Y  |
  | Pure VLA      |  X.X ± Y.Y      | X.X ± Y.Y   | X.X ± Y.Y   | 100%     | X.X ± Y.Y  |
  | Pure A* (PP)  |  X.X ± Y.Y      | X.X ± Y.Y   | X.X ± Y.Y   |  0%      | —           |
  | TIC-VLA (C2)  |  X.X ± Y.Y      | X.X ± Y.Y   | X.X ± Y.Y   | 100%     | X.X ± Y.Y  |  ← optional

Pure VLA:    bare VLA output without gate — vla_* fields from Phase 7 results.
Hybrid:      confidence-gated output — fin_* fields from Phase 7 results.
Pure A*:     A* plan → PurePursuit, compared to same GT futures (no GPU needed).
TIC-VLA:     Phase 9 baseline results (same JSON schema, pass --phase9-dir).

Usage (on Ada after SLURM array completes):
  python3 run_eval_suite.py \\
      --results-dir /home2/om.kathalkar/logs/phase7 \\
      --data-root   /ssd_scratch/om.kathalkar/bw17_dynav \\
      --grid-dir    /ssd_scratch/om.kathalkar/nav_stack_grid \\
      [--phase9-dir /home2/om.kathalkar/logs/phase9] \\
      [--data-root-old /home/cvit-car-simulator/Desktop/bw17_dynav] \\
      [--data-root-new /ssd_scratch/om.kathalkar/bw17_dynav]
"""

import argparse, json, math, os, sys
from pathlib import Path

# nav_stack imports
_NAV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NAV))
from nav_stack.grid.generate_occupancy_grid import build_parametric_grid, inflate_grid, save_grid, load_grid
from nav_stack.planning.astar_planner import plan_path, load_grid_and_meta
from nav_stack.control.pure_pursuit import PurePursuit


# ── Statistics ────────────────────────────────────────────────────────────────

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    n   = len(vals)
    mu  = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


def fmt_ms(mu, sd, digits=2, unit=""):
    if mu is None:
        return "—"
    return f"{mu:.{digits}f} ± {sd:.{digits}f}{unit}"


# ── GT helpers ────────────────────────────────────────────────────────────────

def remap_path(p, old_root, new_root):
    if old_root and new_root and p and p.startswith(old_root):
        return new_root + p[len(old_root):]
    return p


def load_gt_future(window_dir: Path, old_root: str, new_root: str):
    """Return (gt_future list, curr_img path) for the first t_*.json in window."""
    json_files = sorted(window_dir.glob("t_*.json"))
    if not json_files:
        return None, None
    data = json.loads(json_files[0].read_text())
    curr_img = remap_path(data.get("current", {}).get("img", ""), old_root, new_root)
    return data.get("future", []), curr_img


def gt_first_offset(gt_future):
    if not gt_future:
        return None
    off = gt_future[0].get("offset", [0, 0, 0])
    dx, dy = float(off[0]), float(off[1])
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return None
    return dx, dy


def action_to_displacement(lin, ang, dt=0.2):
    fwd = lin * dt
    lat = lin * math.sin(ang * dt) if abs(ang) > 1e-6 else 0.0
    return fwd, lat


def heading_error_deg(pred_dx, pred_dy, gt_dx, gt_dy):
    pred_a = math.atan2(pred_dy, max(abs(pred_dx), 1e-6) * (1 if pred_dx >= 0 else -1))
    gt_a   = math.atan2(gt_dy,   max(abs(gt_dx),   1e-6) * (1 if gt_dx   >= 0 else -1))
    err = abs(pred_a - gt_a) % (2 * math.pi)
    if err > math.pi:
        err = 2 * math.pi - err
    return math.degrees(err)


def ade_fde(pred_dx, pred_dy, gt_future, horizon=10):
    n = min(horizon, len(gt_future))
    if n == 0:
        return None, None
    dists, cx, cy = [], 0.0, 0.0
    for i in range(n):
        cx += pred_dx; cy += pred_dy
        gt = gt_future[i].get("offset", [0, 0, 0])
        dists.append(math.hypot(cx - float(gt[0]), cy - float(gt[1])))
    return sum(dists) / len(dists), dists[-1]


# ── Occupancy grid (cached) ───────────────────────────────────────────────────

_grid_cache = {}


def get_grid(grid_dir: str):
    if grid_dir not in _grid_cache:
        gd = Path(grid_dir)
        if (gd / "warehouse_occupancy_grid.npy").exists():
            grid, meta = load_grid_and_meta(grid_dir)
        else:
            print(f"Generating grid → {grid_dir}", flush=True)
            grid, meta = build_parametric_grid(resolution=0.05)
            grid = inflate_grid(grid, 0.4, 0.05)
            save_grid(grid, meta, gd)
        _grid_cache[grid_dir] = (grid, meta)
    return _grid_cache[grid_dir]


# ── Pure-A* baseline for one window ──────────────────────────────────────────

_pp = PurePursuit(lookahead_dist=1.2, max_lin=0.3, min_lin=0.1)


def compute_pure_astar(start_xy, goal_xy, gt_future, grid_dir):
    """Return (heading_err_deg, ade, fde) for PurePursuit following A* plan."""
    gt_off = gt_first_offset(gt_future)
    if gt_off is None:
        return None, None, None
    gt_dx, gt_dy = gt_off

    grid, meta = get_grid(grid_dir)
    try:
        waypoints = plan_path(grid, tuple(start_xy), tuple(goal_xy), meta)
    except Exception:
        return None, None, None
    if not waypoints:
        return None, None, None

    pp_lin, pp_ang = _pp.compute(start_xy[0], start_xy[1], 0.0, waypoints, 0)
    pp_dx, pp_dy   = action_to_displacement(pp_lin, pp_ang)

    h_err = heading_error_deg(pp_dx, pp_dy, gt_dx, gt_dy)
    a, f  = ade_fde(pp_dx, pp_dy, gt_future)
    return h_err, a, f


# ── Load one set of results ───────────────────────────────────────────────────

def load_results(results_dir: Path):
    files = sorted(results_dir.glob("*_result.json"))
    records = []
    for f in files:
        r = json.loads(f.read_text())
        if r.get("skipped") or r.get("error"):
            continue
        records.append(r)
    return records


# ── Build per-window metrics dict ─────────────────────────────────────────────

def build_metrics(records, data_root, grid_dir, old_root, new_root, label="phase7"):
    """
    For each record extract hybrid, pure-VLA, and pure-A* metrics.
    Returns list of dicts with keys:
      hybrid_h, hybrid_ade, hybrid_fde, hybrid_lat
      vla_h, vla_ade, vla_fde
      pp_h, pp_ade, pp_fde
      used_vla
    """
    results = []
    n = len(records)
    for i, r in enumerate(records, 1):
        print(f"\r  [{label}] {i}/{n}  {r['window']}          ", end="", flush=True)

        # Hybrid & pure-VLA: already computed in Phase 7 result
        entry = {
            "window":   r["window"],
            "used_vla": r.get("used_vla"),
            # Hybrid
            "hybrid_h":   r.get("heading_err_final_deg"),
            "hybrid_ade": r.get("ade_final_m"),
            "hybrid_fde": r.get("fde_final_m"),
            "hybrid_lat": r.get("latency_s"),
            # Pure VLA
            "vla_h":   r.get("heading_err_vla_deg"),
            "vla_ade": r.get("ade_vla_m"),
            "vla_fde": r.get("fde_vla_m"),
            "vla_lat": r.get("latency_s"),
            # Pure A*: computed below
            "pp_h":   None,
            "pp_ade": None,
            "pp_fde": None,
        }

        # Pure A*: reload GT future from window dir, then run PP
        if data_root and grid_dir:
            window_dir = Path(data_root) / "test" / "DynaNav_json" / r["window"]
            if window_dir.exists():
                gt_future, _ = load_gt_future(window_dir, old_root, new_root)
                if gt_future:
                    h, a, f = compute_pure_astar(
                        r.get("start_xy", [3.0, 1.0]),
                        r.get("goal_xy",  [22.0, 15.0]),
                        gt_future, grid_dir,
                    )
                    entry["pp_h"], entry["pp_ade"], entry["pp_fde"] = h, a, f

        results.append(entry)

    print()
    return results


# ── Aggregate one method column ───────────────────────────────────────────────

def agg_column(metrics, h_key, ade_key, fde_key, lat_key=None):
    h_vals   = [m[h_key]   for m in metrics if m[h_key]   is not None]
    ade_vals = [m[ade_key] for m in metrics if m[ade_key] is not None]
    fde_vals = [m[fde_key] for m in metrics if m[fde_key] is not None]
    lat_vals = ([m[lat_key] for m in metrics if m.get(lat_key) is not None]
                if lat_key else [])
    return {
        "n":            len(h_vals),
        "heading_mu":   mean_std(h_vals)[0],
        "heading_sd":   mean_std(h_vals)[1],
        "ade_mu":       mean_std(ade_vals)[0],
        "ade_sd":       mean_std(ade_vals)[1],
        "fde_mu":       mean_std(fde_vals)[0],
        "fde_sd":       mean_std(fde_vals)[1],
        "lat_mu":       mean_std(lat_vals)[0],
        "lat_sd":       mean_std(lat_vals)[1],
    }


# ── Render tables ─────────────────────────────────────────────────────────────

def render_table(rows, out_path: Path):
    """rows: list of (label, vla_rate_str, col_dict)"""
    W = [18, 17, 13, 13, 9, 13]   # column widths
    headers = ["Method", "Heading Err (°)", "ADE (m)", "FDE (m)", "VLA Rate", "Latency (s)"]
    sep = "+" + "+".join("-" * (w + 2) for w in W) + "+"
    hdr = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, W)) + "|"

    lines = [sep, hdr, sep]
    for label, vla_rate, col in rows:
        h_str   = fmt_ms(col["heading_mu"], col["heading_sd"], 2, "°")
        ade_str = fmt_ms(col["ade_mu"],     col["ade_sd"],     4, " m")
        fde_str = fmt_ms(col["fde_mu"],     col["fde_sd"],     4, " m")
        lat_str = fmt_ms(col["lat_mu"],     col["lat_sd"],     2, " s")
        cells = [label, h_str, ade_str, fde_str, vla_rate, lat_str]
        lines.append("|" + "|".join(f" {c:<{w}} " for c, w in zip(cells, W)) + "|")
    lines.append(sep)

    table = "\n".join(lines)
    print(table)
    out_path.write_text(table + "\n")


def render_latex(rows, out_path: Path):
    """LaTeX booktabs table for copy-paste into paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{C3: Hybrid Nav Stack vs Baselines on BW17 DynaNav (N windows)}",
        r"\label{tab:c3_comparison}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Method & Heading Err ($^\circ$) & ADE (m) & FDE (m) & VLA Rate & Latency (s) \\",
        r"\midrule",
    ]
    for label, vla_rate, col in rows:
        h  = fmt_ms(col["heading_mu"], col["heading_sd"], 2)
        a  = fmt_ms(col["ade_mu"],     col["ade_sd"],     4)
        f  = fmt_ms(col["fde_mu"],     col["fde_sd"],     4)
        lt = fmt_ms(col["lat_mu"],     col["lat_sd"],     2) if col["lat_mu"] else r"\textrm{---}"
        lines.append(rf"{label} & {h} & {a} & {f} & {vla_rate} & {lt} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)
    out_path.write_text(latex + "\n")
    print(f"\nLaTeX → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",  required=True,
                    help="Phase 7 SLURM output dir containing *_result.json")
    ap.add_argument("--data-root",    default="",
                    help="BW17 DynaNav root (for GT future + pure-A* baseline)")
    ap.add_argument("--grid-dir",     default="",
                    help="Occupancy grid cache dir (for pure-A* baseline)")
    ap.add_argument("--phase9-dir",   default="",
                    help="Optional Phase 9 TIC-VLA result dir (same JSON schema)")
    ap.add_argument("--data-root-old", default="")
    ap.add_argument("--data-root-new", default="")
    ap.add_argument("--out-dir",      default="",
                    help="Output dir for table files (default: results-dir)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    banner = "=" * 68
    print(banner)
    print("Phase 8 — C3 Comparison Table")
    print(banner)

    # ── Load Phase 7 results ──────────────────────────────────────────────
    print(f"\nLoading Phase 7 results from {results_dir} ...")
    records = load_results(results_dir)
    print(f"  {len(records)} usable windows")

    print("\nComputing metrics (pure-A* requires A* replan + PP — no GPU)...")
    metrics = build_metrics(
        records,
        data_root = args.data_root or None,
        grid_dir  = args.grid_dir  or None,
        old_root  = args.data_root_old,
        new_root  = args.data_root_new,
    )

    n_vla      = sum(1 for m in metrics if m["used_vla"])
    n_fallback = len(metrics) - n_vla
    vla_rate   = n_vla / len(metrics) if metrics else 0.0

    # ── Aggregate columns ─────────────────────────────────────────────────
    hybrid_col = agg_column(metrics, "hybrid_h", "hybrid_ade", "hybrid_fde", "hybrid_lat")
    vla_col    = agg_column(metrics, "vla_h",    "vla_ade",    "vla_fde",    "vla_lat")
    pp_col     = agg_column(metrics, "pp_h",     "pp_ade",     "pp_fde")

    rows = [
        ("Hybrid (ours)", f"{vla_rate:.1%}", hybrid_col),
        ("Pure VLA",      "100%",            vla_col),
        ("Pure A* (PP)",  "0%",              pp_col),
    ]

    # ── Optional Phase 9 TIC-VLA column ──────────────────────────────────
    if args.phase9_dir and Path(args.phase9_dir).is_dir():
        print(f"\nLoading Phase 9 (TIC-VLA) results from {args.phase9_dir} ...")
        p9_records = load_results(Path(args.phase9_dir))
        print(f"  {len(p9_records)} usable windows")
        p9_metrics = build_metrics(
            p9_records,
            data_root = args.data_root or None,
            grid_dir  = None,   # no PP needed for TIC-VLA
            old_root  = args.data_root_old,
            new_root  = args.data_root_new,
            label     = "phase9",
        )
        p9_col = agg_column(p9_metrics, "vla_h", "vla_ade", "vla_fde", "vla_lat")
        rows.append(("TIC-VLA (C2)", "100%", p9_col))

    # ── Print and save ────────────────────────────────────────────────────
    print(f"\n{banner}")
    print("C3 Comparison Table")
    print(banner)
    print(f"  N windows: {len(metrics)}  |  VLA: {n_vla} ({vla_rate:.1%})  |  PP: {n_fallback}")
    print()
    render_table(rows, out_dir / "eval_comparison.txt")
    render_latex(rows, out_dir / "eval_comparison.tex")

    # ── Save JSON ─────────────────────────────────────────────────────────
    def col_to_dict(col):
        return {
            "n":            col["n"],
            "heading_err_deg": {"mean": col["heading_mu"], "std": col["heading_sd"]},
            "ade_m":           {"mean": col["ade_mu"],     "std": col["ade_sd"]},
            "fde_m":           {"mean": col["fde_mu"],     "std": col["fde_sd"]},
            "latency_s":       {"mean": col["lat_mu"],     "std": col["lat_sd"]},
        }

    output = {
        "n_windows":    len(metrics),
        "vla_rate":     round(vla_rate, 4),
        "fallback_rate": round(1 - vla_rate, 4),
        "hybrid":       col_to_dict(hybrid_col),
        "pure_vla":     col_to_dict(vla_col),
        "pure_astar":   col_to_dict(pp_col),
    }
    if args.phase9_dir and Path(args.phase9_dir).is_dir():
        output["ticvla_c2"] = col_to_dict(p9_col)

    json_path = out_dir / "eval_comparison.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nJSON   → {json_path}")
    print(f"Text   → {out_dir / 'eval_comparison.txt'}")
    print(f"LaTeX  → {out_dir / 'eval_comparison.tex'}")
    print(banner)


if __name__ == "__main__":
    main()
