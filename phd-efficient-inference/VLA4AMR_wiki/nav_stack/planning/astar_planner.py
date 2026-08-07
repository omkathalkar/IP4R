#!/usr/bin/env python3
"""
Phase 2 — A* global planner for the VLA4AMR hybrid nav stack.

Usage:
  from nav_stack.planning.astar_planner import plan_path, load_grid_and_meta
  waypoints = plan_path(grid, (start_x, start_y), (goal_x, goal_y), meta)

CLI (for visual inspection):
  python3 astar_planner.py --grid-dir /tmp/grid \
      --start 2.5 3.0 --goal 22.0 14.0 --plot
"""

import sys
import math
import heapq
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GridMeta, Waypoint


# ── A* core ───────────────────────────────────────────────────────────────────

def _octile(c0: int, r0: int, c1: int, r1: int) -> float:
    """Octile distance heuristic — admissible for 8-connected grids."""
    dc, dr = abs(c1 - c0), abs(r1 - r0)
    return max(dc, dr) + (math.sqrt(2) - 1) * min(dc, dr)


def astar_cells(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    """
    8-connected A* on a binary occupancy grid.
    grid: uint8, 0=free, 255=occupied (after inflation — call inflate_grid first).
    start/goal: (col, row) cell indices.
    Returns list of (col, row) from start to goal, or None if no path found.
    """
    rows, cols = grid.shape
    sc, sr = start
    gc, gr = goal

    if grid[sr, sc] > 127:
        raise ValueError(f"Start cell ({sc},{sr}) is inside an obstacle.")
    if grid[gr, gc] > 127:
        raise ValueError(f"Goal cell ({gc},{gr}) is inside an obstacle.")

    MOVES = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    COSTS = [1.0, 1.0, 1.0, 1.0, math.sqrt(2)] * 2  # diagonal costs

    g_score = np.full((rows, cols), np.inf)
    g_score[sr, sc] = 0.0
    came_from: dict[tuple[int,int], tuple[int,int]] = {}

    # (f, g, col, row)
    open_heap = [(0.0 + _octile(sc, sr, gc, gr), 0.0, sc, sr)]

    while open_heap:
        f, g, c, r = heapq.heappop(open_heap)

        if (c, r) == (gc, gr):
            # Reconstruct
            path = [(c, r)]
            while (c, r) in came_from:
                c, r = came_from[(c, r)]
                path.append((c, r))
            path.reverse()
            return path

        if g > g_score[r, c] + 1e-9:
            continue  # stale entry

        for i, (dc, dr) in enumerate(MOVES):
            nc, nr = c + dc, r + dr
            if not (0 <= nc < cols and 0 <= nr < rows):
                continue
            if grid[nr, nc] > 127:
                continue
            step = math.sqrt(2) if (dc != 0 and dr != 0) else 1.0
            ng = g + step
            if ng < g_score[nr, nc] - 1e-9:
                g_score[nr, nc] = ng
                came_from[(nc, nr)] = (c, r)
                heapq.heappush(open_heap, (ng + _octile(nc, nr, gc, gr), ng, nc, nr))

    return None  # no path


# ── Path post-processing ───────────────────────────────────────────────────────

def douglas_peucker(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker line simplification."""
    if len(points) < 3:
        return points

    def perpendicular_dist(px, py, ax, ay, bx, by) -> float:
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    dmax, idx = 0.0, 0
    ax, ay = points[0]
    bx, by = points[-1]
    for i in range(1, len(points) - 1):
        d = perpendicular_dist(points[i][0], points[i][1], ax, ay, bx, by)
        if d > dmax:
            dmax, idx = d, i

    if dmax > epsilon:
        left  = douglas_peucker(points[:idx + 1], epsilon)
        right = douglas_peucker(points[idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


def enforce_min_spacing(
    pts: list[tuple[float, float]], min_dist: float
) -> list[tuple[float, float]]:
    """Remove intermediate points closer than min_dist to the previous kept point."""
    if not pts:
        return pts
    kept = [pts[0]]
    for p in pts[1:-1]:
        if math.hypot(p[0] - kept[-1][0], p[1] - kept[-1][1]) >= min_dist:
            kept.append(p)
    kept.append(pts[-1])
    return kept


def cells_to_waypoints(
    cell_path: list[tuple[int, int]],
    meta: GridMeta,
    dp_epsilon: float = 0.15,
    min_spacing: float = 0.5,
) -> list[Waypoint]:
    """
    Convert a raw A* cell path to a list of Waypoints in world coordinates.
    Steps:
      1. Cell centres → world (x, y)
      2. Douglas-Peucker simplification (epsilon in metres)
      3. Enforce minimum inter-waypoint spacing
      4. Inject headings: θ[i] = atan2(y[i+1]-y[i], x[i+1]-x[i]),
         last waypoint inherits heading of previous segment
    """
    # Step 1: cells → world
    world = [meta.cell_to_world(c, r) for c, r in cell_path]

    # Step 2: simplify
    simplified = douglas_peucker(world, dp_epsilon)

    # Step 3: min spacing
    spaced = enforce_min_spacing(simplified, min_spacing)

    if len(spaced) < 2:
        # Degenerate path (start == goal)
        wx, wy = spaced[0]
        return [Waypoint(wx, wy, 0.0)]

    # Step 4: headings
    waypoints = []
    for i in range(len(spaced) - 1):
        x0, y0 = spaced[i]
        x1, y1 = spaced[i + 1]
        theta = math.atan2(y1 - y0, x1 - x0)
        waypoints.append(Waypoint(x0, y0, theta))
    # Last waypoint: same heading as penultimate segment
    waypoints.append(Waypoint(spaced[-1][0], spaced[-1][1], waypoints[-1].theta))

    return waypoints


# ── Public interface ───────────────────────────────────────────────────────────

def plan_path(
    grid: np.ndarray,
    start_world: tuple[float, float],
    goal_world: tuple[float, float],
    meta: GridMeta,
    dp_epsilon: float = 0.15,
    min_spacing: float = 0.5,
) -> list[Waypoint]:
    """
    Plan a path from start_world to goal_world over the occupancy grid.

    grid should already be inflated (from inflate_grid) before calling this.

    Returns list of Waypoint(x, y, theta) in world frame.
    Raises ValueError if start/goal is in obstacle, or RuntimeError if no path found.
    """
    sc, sr = meta.world_to_cell(*start_world)
    gc, gr = meta.world_to_cell(*goal_world)

    cell_path = astar_cells(grid, (sc, sr), (gc, gr))
    if cell_path is None:
        raise RuntimeError(
            f"A* found no path from world {start_world} (cell {sc},{sr}) "
            f"to {goal_world} (cell {gc},{gr}). "
            "Check that the grid is inflated and start/goal are in free space."
        )

    return cells_to_waypoints(cell_path, meta, dp_epsilon=dp_epsilon, min_spacing=min_spacing)


# ── Grid I/O helper ───────────────────────────────────────────────────────────

def load_grid_and_meta(grid_dir: str | Path) -> tuple[np.ndarray, GridMeta]:
    import yaml
    d = Path(grid_dir)
    grid = np.load(d / "warehouse_occupancy_grid.npy")
    meta_dict = yaml.safe_load((d / "warehouse_occupancy_meta.yaml").read_text())
    return grid, GridMeta(**meta_dict)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_plan(
    grid: np.ndarray,
    meta: GridMeta,
    waypoints: list[Waypoint],
    start_world: tuple[float, float],
    goal_world:  tuple[float, float],
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [
        meta.origin_x, meta.origin_x + meta.width  * meta.resolution,
        meta.origin_y, meta.origin_y + meta.height * meta.resolution,
    ]
    ax.imshow(np.flipud(grid), cmap="Greys", extent=extent, origin="upper", vmin=0, vmax=255, alpha=0.6)

    # Waypoint path
    xs = [w.x for w in waypoints]
    ys = [w.y for w in waypoints]
    ax.plot(xs, ys, "b-o", markersize=5, linewidth=1.8, label=f"A* path ({len(waypoints)} WPs)", zorder=3)

    # Heading arrows
    for wp in waypoints:
        ax.annotate("", xy=(wp.x + 0.3 * math.cos(wp.theta), wp.y + 0.3 * math.sin(wp.theta)),
                    xytext=(wp.x, wp.y),
                    arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.2))

    ax.plot(*start_world, "gs", markersize=12, zorder=5, label="start")
    ax.plot(*goal_world,  "r*", markersize=14, zorder=5, label="goal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"A* plan — {len(waypoints)} waypoints, spacing ≥ 0.5m")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plan plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="A* path planner for VLA4AMR")
    ap.add_argument("--grid-dir", required=True, help="Directory with grid .npy + meta .yaml")
    ap.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"), required=True)
    ap.add_argument("--goal",  nargs=2, type=float, metavar=("X", "Y"), required=True)
    ap.add_argument("--min-spacing", type=float, default=0.5, help="Min waypoint spacing (m)")
    ap.add_argument("--dp-epsilon",  type=float, default=0.15, help="Douglas-Peucker epsilon (m)")
    ap.add_argument("--plot",   action="store_true")
    ap.add_argument("--out-dir", default=None, help="Save plan PNG here (default: grid-dir)")
    args = ap.parse_args()

    grid, meta = load_grid_and_meta(args.grid_dir)
    print(f"Loaded grid {meta.width}×{meta.height} @ {meta.resolution}m/cell")

    start = tuple(args.start)
    goal  = tuple(args.goal)

    waypoints = plan_path(grid, start, goal, meta,
                          dp_epsilon=args.dp_epsilon, min_spacing=args.min_spacing)

    print(f"Path: {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i:02d}  x={wp.x:7.3f}  y={wp.y:7.3f}  θ={math.degrees(wp.theta):+7.1f}°")

    if args.plot:
        out_dir = Path(args.out_dir or args.grid_dir)
        plot_plan(grid, meta, waypoints, start, goal, save_path=out_dir / "plan_preview.png")


if __name__ == "__main__":
    main()
