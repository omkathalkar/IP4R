#!/usr/bin/env python3
"""
Phase 1 — Occupancy grid generation for the VLA4AMR hybrid nav stack.

Two backends, tried in order:
  1. Isaac Sim omap extension (isaacsim.asset.gen.omap) — accurate, requires
     a running Isaac Sim session with GPU-accelerated X.
  2. Parametric warehouse hand-rasterizer — no Isaac Sim dependency; uses
     configurable aisle/rack layout that approximates carter_warehouse_navigation.usd.

Both backends produce the same outputs:
  warehouse_occupancy_grid.npy   — uint8 array, 0=free, 255=occupied
  warehouse_occupancy_meta.yaml  — GridMeta fields (resolution, origin, size)

Usage:
  # Parametric (no Isaac Sim):
  python3 generate_occupancy_grid.py --backend parametric --out-dir /tmp/grid

  # Isaac Sim omap (run from inside an Isaac Sim Python environment):
  python3 generate_occupancy_grid.py --backend omap --scene-url omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/carter_warehouse_navigation.usd
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path

# Local import (nav_stack root must be in PYTHONPATH or called from there)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GridMeta


# ── Parametric backend ─────────────────────────────────────────────────────────

def _draw_rect(grid: np.ndarray, col0: int, row0: int, col1: int, row1: int):
    """Mark a rectangle of cells as occupied (255). Clips to grid bounds."""
    h, w = grid.shape
    r0, r1 = max(0, row0), min(h, row1)
    c0, c1 = max(0, col0), min(w, col1)
    grid[r0:r1, c0:c1] = 255


def _w2c(val: float, origin: float, res: float) -> int:
    return int((val - origin) / res)


def build_parametric_grid(
    resolution: float = 0.05,
    arena_w: float = 30.0,
    arena_h: float = 20.0,
    origin_x: float = -2.0,
    origin_y: float = -2.0,
    wall_thickness: float = 0.3,
    rack_depth: float = 1.2,
    rack_gap: float = 3.0,
    aisle_width: float = 2.5,
    rack_length: float = 8.0,
    num_rack_rows: int = 3,
    rack_y_start: float = 2.0,
    rack_x_start: float = 2.0,
) -> tuple[np.ndarray, GridMeta]:
    """
    Hand-rasterized approximation of carter_warehouse_navigation.usd.

    Layout (all dimensions in world metres, Y = forward):
      - Outer walls on all four sides.
      - num_rack_rows rows of paired shelving units (each pair: rack + aisle + rack).
        Racks run parallel to the X axis (i.e. they extend in X, separated in Y).

    Tune parameters to match the actual scene by comparing the plotted grid
    against the IsaacSim top-down view of carter_warehouse_navigation.usd.
    """
    cols = int(arena_w / resolution)
    rows = int(arena_h / resolution)
    grid = np.zeros((rows, cols), dtype=np.uint8)

    def wx(x): return _w2c(x, origin_x, resolution)
    def wy(y): return _w2c(y, origin_y, resolution)
    def dw(d): return max(1, int(d / resolution))

    # Outer walls
    _draw_rect(grid, 0,           0,            cols, dw(wall_thickness))   # south
    _draw_rect(grid, 0,           rows - dw(wall_thickness), cols, rows)    # north
    _draw_rect(grid, 0,           0,            dw(wall_thickness), rows)   # west
    _draw_rect(grid, cols - dw(wall_thickness), 0, cols, rows)              # east

    # Rack rows — each row: two parallel racks separated by an aisle
    y_cursor = rack_y_start
    for _ in range(num_rack_rows):
        # First rack of the pair
        _draw_rect(grid,
                   wx(rack_x_start), wy(y_cursor),
                   wx(rack_x_start + rack_length), wy(y_cursor + rack_depth))
        y_cursor += rack_depth + aisle_width
        # Second rack of the pair
        _draw_rect(grid,
                   wx(rack_x_start), wy(y_cursor),
                   wx(rack_x_start + rack_length), wy(y_cursor + rack_depth))
        y_cursor += rack_depth + rack_gap

    meta = GridMeta(
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        width=cols,
        height=rows,
    )
    return grid, meta


# ── Isaac Sim omap backend ─────────────────────────────────────────────────────

def build_omap_grid(
    scene_url: str,
    resolution: float = 0.05,
    height_min: float = 0.1,
    height_max: float = 2.0,
) -> tuple[np.ndarray, GridMeta]:
    """
    Use Isaac Sim's built-in Occupancy Map Generator extension to extract
    a 2D binary grid from the warehouse USD.

    Must be called from within an active Isaac Sim Python environment
    (isaacsim.asset.gen.omap must be available).

    height_min/height_max: only geometry between these heights (metres) is
    considered for occupancy (avoids floor and ceiling artefacts).
    """
    try:
        from isaacsim import SimulationApp
    except ImportError:
        raise RuntimeError(
            "isaacsim not found — run this script from the Isaac Sim Python environment, "
            "or use --backend parametric."
        )

    app = SimulationApp({"headless": True, "active_gpu": 0})

    import omni.kit.app
    from omni.isaac.core.utils.stage import open_stage
    import omni.isaac.occupancy_map as omap_ext

    open_stage(scene_url)
    # Wait for scene to load fully
    for _ in range(30):
        app.update()

    generator = omap_ext.OccupancyMapGenerator()
    generator.update_settings(cell_size=resolution, height_min=height_min, height_max=height_max)
    generator.compute_occupancy_map()

    grid_raw = generator.get_buffer()           # flat uint8: 0=free, 100=occupied
    w, h     = generator.get_dimensions()
    origin   = generator.get_origin()           # (x, y) world of cell (0,0)

    grid = (grid_raw.reshape(h, w) > 50).astype(np.uint8) * 255

    meta = GridMeta(
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        width=w,
        height=h,
    )

    app.close()
    return grid, meta


# ── Inflation ──────────────────────────────────────────────────────────────────

def inflate_grid(grid: np.ndarray, robot_radius_m: float, resolution: float) -> np.ndarray:
    """
    Inflate obstacles by robot_radius to create a configuration-space grid.
    Any cell within robot_radius of an obstacle becomes occupied.
    Uses binary dilation via scipy.
    """
    from scipy.ndimage import binary_dilation
    radius_cells = int(np.ceil(robot_radius_m / resolution))
    struct = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=bool)
    occupied = grid > 127
    inflated = binary_dilation(occupied, structure=struct)
    return inflated.astype(np.uint8) * 255


# ── Sanity check ──────────────────────────────────────────────────────────────

def flood_fill_check(grid: np.ndarray, start_cell: tuple[int, int], goal_cell: tuple[int, int]) -> bool:
    """
    BFS from start_cell to check goal_cell is reachable in the free space.
    Returns True if connected, False otherwise.
    """
    from collections import deque
    rows, cols = grid.shape
    sc, sr = start_cell
    gc, gr = goal_cell

    if grid[sr, sc] > 127 or grid[gr, gc] > 127:
        return False

    visited = np.zeros((rows, cols), dtype=bool)
    visited[sr, sc] = True
    q: deque[tuple[int, int]] = deque([(sc, sr)])
    while q:
        c, r = q.popleft()
        if (c, r) == (gc, gr):
            return True
        for dc, dr in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and not visited[nr, nc] and grid[nr, nc] <= 127:
                visited[nr, nc] = True
                q.append((nc, nr))
    return False


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_grid(grid: np.ndarray, meta: GridMeta, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "warehouse_occupancy_grid.npy", grid)
    meta_dict = {
        "resolution": meta.resolution,
        "origin_x": meta.origin_x,
        "origin_y": meta.origin_y,
        "width": meta.width,
        "height": meta.height,
    }
    (out_dir / "warehouse_occupancy_meta.yaml").write_text(yaml.dump(meta_dict))
    print(f"Saved grid {meta.width}×{meta.height} @ {meta.resolution}m/cell to {out_dir}")


def load_grid(out_dir: Path) -> tuple[np.ndarray, GridMeta]:
    grid = np.load(out_dir / "warehouse_occupancy_grid.npy")
    meta_dict = yaml.safe_load((out_dir / "warehouse_occupancy_meta.yaml").read_text())
    meta = GridMeta(**meta_dict)
    return grid, meta


def load_pgm(pgm_path: Path, yaml_path: Path) -> tuple[np.ndarray, GridMeta]:
    """Load a ROS-format occupancy map (PGM + YAML)."""
    import cv2
    raw = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Cannot read {pgm_path}")
    # ROS convention: 0=occupied (black), 205=unknown (grey), 254=free (white)
    occupied = raw < 128
    grid = occupied.astype(np.uint8) * 255

    ros = yaml.safe_load(yaml_path.read_text())
    ox, oy = ros["origin"][0], ros["origin"][1]
    res = float(ros["resolution"])
    h, w = grid.shape
    meta = GridMeta(resolution=res, origin_x=ox, origin_y=oy, width=w, height=h)
    return grid, meta


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_grid(grid: np.ndarray, meta: GridMeta, title: str = "Occupancy grid",
              waypoints=None, save_path: Path | None = None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 7))
    extent = [
        meta.origin_x, meta.origin_x + meta.width  * meta.resolution,
        meta.origin_y, meta.origin_y + meta.height * meta.resolution,
    ]
    # Flip grid vertically so Y increases upward
    ax.imshow(np.flipud(grid), cmap="binary", extent=extent, origin="upper", vmin=0, vmax=255)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)

    if waypoints:
        xs = [w.x for w in waypoints]
        ys = [w.y for w in waypoints]
        ax.plot(xs, ys, "b-o", markersize=4, linewidth=1.5, label="waypoints")
        ax.plot(xs[0],  ys[0],  "gs", markersize=9, zorder=5, label="start")
        ax.plot(xs[-1], ys[-1], "r*", markersize=12, zorder=5, label="goal")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grid plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate warehouse occupancy grid")
    ap.add_argument("--backend", choices=["parametric", "omap", "pgm"], default="parametric")
    ap.add_argument("--out-dir", default="grid_output")
    ap.add_argument("--resolution", type=float, default=0.05, help="metres per cell")
    ap.add_argument("--robot-radius", type=float, default=0.4,
                    help="Robot inflation radius in metres (NovaCarter ≈ 0.4m)")
    ap.add_argument("--plot", action="store_true", help="Show/save a visualisation")
    # omap-specific
    ap.add_argument("--scene-url", default="",
                    help="USD URL for omap backend")
    # pgm-specific
    ap.add_argument("--pgm", default="", help="Path to .pgm file")
    ap.add_argument("--pgm-yaml", default="", help="Path to matching .yaml file")
    # parametric tuning
    ap.add_argument("--arena-w",   type=float, default=30.0)
    ap.add_argument("--arena-h",   type=float, default=20.0)
    ap.add_argument("--origin-x",  type=float, default=-2.0)
    ap.add_argument("--origin-y",  type=float, default=-2.0)
    # flood-fill sanity check
    ap.add_argument("--check-start", nargs=2, type=float, metavar=("X", "Y"),
                    help="World coords of start for flood-fill check")
    ap.add_argument("--check-goal",  nargs=2, type=float, metavar=("X", "Y"),
                    help="World coords of goal for flood-fill check")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.backend == "omap":
        if not args.scene_url:
            ap.error("--scene-url required for omap backend")
        grid, meta = build_omap_grid(args.scene_url, args.resolution)
    elif args.backend == "pgm":
        if not args.pgm or not args.pgm_yaml:
            ap.error("--pgm and --pgm-yaml required for pgm backend")
        grid, meta = load_pgm(Path(args.pgm), Path(args.pgm_yaml))
    else:
        grid, meta = build_parametric_grid(
            resolution=args.resolution,
            arena_w=args.arena_w,
            arena_h=args.arena_h,
            origin_x=args.origin_x,
            origin_y=args.origin_y,
        )

    print(f"Raw grid: {meta.width}×{meta.height} cells, "
          f"{grid.sum()//255} occupied / {grid.size} total")

    inflated = inflate_grid(grid, args.robot_radius, meta.resolution)
    print(f"Inflated (r={args.robot_radius}m): {inflated.sum()//255} occupied cells")

    if args.check_start and args.check_goal:
        sc = meta.world_to_cell(*args.check_start)
        gc = meta.world_to_cell(*args.check_goal)
        ok = flood_fill_check(inflated, sc, gc)
        print(f"Flood-fill connectivity {args.check_start} → {args.check_goal}: {'OK' if ok else 'DISCONNECTED'}")
        if not ok:
            print("WARNING: start and goal are not connected in the inflated grid. "
                  "A* will fail. Check layout parameters.")

    save_grid(inflated, meta, out_dir)

    if args.plot:
        plot_grid(inflated, meta,
                  title=f"Warehouse occupancy (inflated r={args.robot_radius}m, {meta.resolution}m/cell)",
                  save_path=out_dir / "grid_preview.png")


if __name__ == "__main__":
    main()
