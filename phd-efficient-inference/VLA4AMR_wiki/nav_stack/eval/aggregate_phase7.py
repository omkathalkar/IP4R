#!/usr/bin/env python3
"""
nav_stack/eval/aggregate_phase7.py — Aggregate Phase 7 per-window results.

Reads all <window>_result.json files from the SLURM array output dir and
prints a summary table suitable for the paper (Table 1, C3 column).

Usage:
  python3 aggregate_phase7.py --results-dir /home2/om.kathalkar/logs/phase7

Output (stdout + phase7_summary.json in results-dir):
  N windows evaluated, k skipped/errored
  VLA rate, fallback rate
  Mean ± std: heading_err_final, ade_final, fde_final
  Mean ± std: heading_err_vla, ade_vla, fde_vla   (pure-VLA baseline row)
  Mean latency
"""

import argparse, json, math, sys
from pathlib import Path


def mean_std(vals):
    if not vals:
        return None, None
    n = len(vals)
    mu = sum(vals) / n
    if n == 1:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


def fmt(mu, sd, unit=""):
    if mu is None:
        return "n/a"
    return f"{mu:.4f} ± {sd:.4f}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    result_files = sorted(rdir.glob("*_result.json"))

    if not result_files:
        print(f"No *_result.json files found in {rdir}", file=sys.stderr)
        sys.exit(1)

    total = len(result_files)
    skipped = errored = vla_count = pp_count = 0

    heading_final, ade_final, fde_final = [], [], []
    heading_vla,   ade_vla,   fde_vla   = [], [], []
    latencies = []
    vla_magnitudes = []

    for f in result_files:
        r = json.loads(f.read_text())

        if r.get("skipped") or r.get("error"):
            if r.get("error"):
                errored += 1
            else:
                skipped += 1
            continue

        if r.get("used_vla"):
            vla_count += 1
        else:
            pp_count += 1

        if r.get("vla_magnitude") is not None:
            vla_magnitudes.append(r["vla_magnitude"])

        if r.get("heading_err_final_deg") is not None:
            heading_final.append(r["heading_err_final_deg"])
        if r.get("ade_final_m") is not None:
            ade_final.append(r["ade_final_m"])
        if r.get("fde_final_m") is not None:
            fde_final.append(r["fde_final_m"])

        if r.get("heading_err_vla_deg") is not None:
            heading_vla.append(r["heading_err_vla_deg"])
        if r.get("ade_vla_m") is not None:
            ade_vla.append(r["ade_vla_m"])
        if r.get("fde_vla_m") is not None:
            fde_vla.append(r["fde_vla_m"])

        if r.get("latency_s") is not None:
            latencies.append(r["latency_s"])

    evaluated = total - skipped - errored
    vla_rate  = vla_count / evaluated if evaluated else 0.0
    pp_rate   = pp_count  / evaluated if evaluated else 0.0

    hf_mu, hf_sd   = mean_std(heading_final)
    af_mu, af_sd   = mean_std(ade_final)
    ff_mu, ff_sd   = mean_std(fde_final)
    hv_mu, hv_sd   = mean_std(heading_vla)
    av_mu, av_sd   = mean_std(ade_vla)
    fv_mu, fv_sd   = mean_std(fde_vla)
    lat_mu, lat_sd = mean_std(latencies)
    mag_mu, mag_sd = mean_std(vla_magnitudes)

    banner = "=" * 62
    print(banner)
    print("Phase 7 — Hybrid Nav Stack Eval Summary")
    print(banner)
    print(f"  Results dir : {rdir}")
    print(f"  Total files : {total}")
    print(f"  Evaluated   : {evaluated}  (skipped={skipped}, errored={errored})")
    print()
    print("  ── VLA / Fallback split ────────────────────────────────")
    print(f"  VLA used    : {vla_count:3d} / {evaluated}  ({vla_rate:.1%})")
    print(f"  Pure-pursuit: {pp_count:3d} / {evaluated}  ({pp_rate:.1%})")
    print(f"  VLA mag     : {fmt(mag_mu, mag_sd)}")
    print()
    print("  ── Hybrid (gate output) ────────────────────────────────")
    print(f"  Heading err : {fmt(hf_mu, hf_sd, '°')}")
    print(f"  ADE         : {fmt(af_mu, af_sd, ' m')}")
    print(f"  FDE         : {fmt(ff_mu, ff_sd, ' m')}")
    print()
    print("  ── Pure VLA (no gate) ──────────────────────────────────")
    print(f"  Heading err : {fmt(hv_mu, hv_sd, '°')}")
    print(f"  ADE         : {fmt(av_mu, av_sd, ' m')}")
    print(f"  FDE         : {fmt(fv_mu, fv_sd, ' m')}")
    print()
    print(f"  Latency     : {fmt(lat_mu, lat_sd, ' s')}")
    print(banner)

    summary = {
        "n_total":        total,
        "n_evaluated":    evaluated,
        "n_skipped":      skipped,
        "n_errored":      errored,
        "vla_rate":       round(vla_rate, 4),
        "fallback_rate":  round(pp_rate,  4),
        "vla_magnitude_mean": round(mag_mu, 4) if mag_mu is not None else None,
        "hybrid": {
            "heading_err_deg_mean": round(hf_mu, 4) if hf_mu is not None else None,
            "heading_err_deg_std":  round(hf_sd, 4) if hf_sd is not None else None,
            "ade_m_mean":           round(af_mu, 4) if af_mu is not None else None,
            "ade_m_std":            round(af_sd, 4) if af_sd is not None else None,
            "fde_m_mean":           round(ff_mu, 4) if ff_mu is not None else None,
            "fde_m_std":            round(ff_sd, 4) if ff_sd is not None else None,
        },
        "pure_vla": {
            "heading_err_deg_mean": round(hv_mu, 4) if hv_mu is not None else None,
            "heading_err_deg_std":  round(hv_sd, 4) if hv_sd is not None else None,
            "ade_m_mean":           round(av_mu, 4) if av_mu is not None else None,
            "ade_m_std":            round(av_sd, 4) if av_sd is not None else None,
            "fde_m_mean":           round(fv_mu, 4) if fv_mu is not None else None,
            "fde_m_std":            round(fv_sd, 4) if fv_sd is not None else None,
        },
        "latency_s_mean": round(lat_mu, 4) if lat_mu is not None else None,
    }

    out_path = rdir / "phase7_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary JSON → {out_path}")


if __name__ == "__main__":
    main()
