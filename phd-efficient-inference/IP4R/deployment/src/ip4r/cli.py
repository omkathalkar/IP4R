"""Command-line entry point: `ip4r <subcommand>`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .config import Config
from .pipeline import Inspector
from .report import write_report
from .roi import load_rois
from .synth import make_defect


def _print_summary(result) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n=== IP4R {status} : {Path(result.image_path).name} ===")
    print(f"registration: {result.registration}")
    for r in result.roi_results:
        flag = "ok " if r.passed else "FAIL"
        cov = f"cov={r.coverage}" if r.coverage is not None else ""
        ssm = f"ssim={r.ssim}" if r.ssim is not None else ""
        extra = f"  <- {r.reason}" if r.reason else ""
        print(f"  [{flag}] {r.name:24s} {cov:>12s} {ssm:>12s}{extra}")
    print(f"  -> {len(result.failed_rois)}/{len(result.roi_results)} element(s) failed\n")


def cmd_selfcheck(cfg: Config, args) -> int:
    """Inspect the golden against itself: should be all-PASS (validates the pipeline)."""
    insp = Inspector(cfg)
    result = insp.inspect_array(insp.golden_bgr.copy(), str(cfg.path("paths.reference_image")))
    _print_summary(result)
    out = write_report(result, insp.golden_bgr, cfg, cfg.path("paths.results_dir"))
    print("written:", out)
    return 0 if result.passed else 2


def cmd_inspect(cfg: Config, args) -> int:
    insp = Inspector(cfg)
    paths = [Path(args.image)] if Path(args.image).is_file() else sorted(Path(args.image).glob("*"))
    rc = 0
    for p in paths:
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        result = insp.inspect_path(p)
        _print_summary(result)
        write_report(result, insp.golden_bgr, cfg, cfg.path("paths.results_dir"))
        if not result.passed:
            rc = 2
    return rc


def cmd_synth(cfg: Config, args) -> int:
    insp_ref = cv2.imread(str(cfg.path("paths.reference_image")), cv2.IMREAD_COLOR)
    rois = load_rois(cfg.path("paths.roi_map"))
    img, desc = make_defect(insp_ref, rois, mode=args.mode, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"synthesised defect ({desc}) -> {out}")
    return 0


def cmd_roi_edit(cfg: Config, args) -> int:
    from .tools.roi_editor import run_editor
    return run_editor(cfg)


def cmd_camera(cfg: Config, args) -> int:
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "inspect_camera",
        Path(__file__).parent.parent.parent.parent / "scripts" / "inspect_camera.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_camera(
        camera_index=args.camera,
        stable_needed=args.stable_frames,
        show_window=not args.no_window,
        re_register_interval=args.re_register,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ip4r", description="AC-remote LCD splash-screen QC")
    p.add_argument("--config", default=None, help="path to config YAML")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selfcheck", help="inspect golden vs itself (sanity)")

    pi = sub.add_parser("inspect", help="inspect an image or a folder")
    pi.add_argument("image", help="image file or directory of images")

    ps = sub.add_parser("synth", help="make a synthetic defective sample")
    ps.add_argument("--mode", default="erase", choices=["erase", "dim", "blur", "shift"])
    ps.add_argument("--out", default="data/samples/defect_01.jpg")
    ps.add_argument("--seed", type=int, default=None)

    sub.add_parser("roi-edit", help="interactively author the ROI map on the golden")

    pc = sub.add_parser("camera", help="live camera inspection (splash-triggered, real-time)")
    pc.add_argument("--camera", type=int, default=0,
                    help="camera device index (0=built-in, 1/2=USB)")
    pc.add_argument("--stable-frames", type=int, default=3,
                    help="consecutive passing frames before triggering inspection (default 3)")
    pc.add_argument("--no-window", action="store_true",
                    help="headless mode: no display, save results only")
    pc.add_argument("--re-register", type=int, default=30,
                    help="re-run registration every N frames (default 30)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load(args.config)
    dispatch = {
        "selfcheck": cmd_selfcheck,
        "inspect": cmd_inspect,
        "synth": cmd_synth,
        "roi-edit": cmd_roi_edit,
        "camera": cmd_camera,
    }
    return dispatch[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
