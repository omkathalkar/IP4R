"""Unit tests for Phase 1 (grid) + Phase 2 (A* planner)."""
import sys
import math
import numpy as np
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import GridMeta, Waypoint
from grid.generate_occupancy_grid import (
    build_parametric_grid, inflate_grid, flood_fill_check
)
from planning.astar_planner import (
    astar_cells, cells_to_waypoints, plan_path,
    douglas_peucker, enforce_min_spacing,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_room(w=20, h=20) -> tuple[np.ndarray, GridMeta]:
    """Fully free grid with walls on the border."""
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[0, :] = 255; grid[-1, :] = 255
    grid[:, 0] = 255; grid[:, -1] = 255
    meta = GridMeta(resolution=1.0, origin_x=0.0, origin_y=0.0, width=w, height=h)
    return grid, meta


def room_with_obstacle(w=20, h=20) -> tuple[np.ndarray, GridMeta]:
    """Open room with a vertical wall obstacle down the middle."""
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[0, :] = 255; grid[-1, :] = 255
    grid[:, 0] = 255; grid[:, -1] = 255
    # Vertical wall at col=10, with a gap at row 15
    grid[2:14, 10] = 255
    grid[16:h-1, 10] = 255
    meta = GridMeta(resolution=1.0, origin_x=0.0, origin_y=0.0, width=w, height=h)
    return grid, meta


# ── GridMeta ──────────────────────────────────────────────────────────────────

def test_gridmeta_roundtrip():
    meta = GridMeta(resolution=0.05, origin_x=-1.0, origin_y=-2.0, width=100, height=80)
    wx, wy = meta.cell_to_world(5, 10)
    c, r = meta.world_to_cell(wx, wy)
    assert c == 5 and r == 10


def test_gridmeta_outside_raises():
    meta = GridMeta(resolution=0.05, origin_x=0.0, origin_y=0.0, width=100, height=100)
    with pytest.raises(ValueError):
        meta.world_to_cell(999.0, 0.0)


# ── Flood fill ────────────────────────────────────────────────────────────────

def test_flood_fill_connected():
    grid, meta = open_room()
    assert flood_fill_check(grid, (1, 1), (18, 18))


def test_flood_fill_disconnected():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 5] = 255   # wall splits left/right
    assert not flood_fill_check(grid, (1, 5), (8, 5))


def test_flood_fill_blocked_start():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[3, 3] = 255
    assert not flood_fill_check(grid, (3, 3), (8, 8))


# ── A* cell path ─────────────────────────────────────────────────────────────

def test_astar_straight():
    grid, _ = open_room(20, 20)
    path = astar_cells(grid, (1, 1), (18, 1))
    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (18, 1)
    # All cells must be free
    for c, r in path:
        assert grid[r, c] == 0


def test_astar_diagonal():
    grid, _ = open_room(20, 20)
    path = astar_cells(grid, (1, 1), (18, 18))
    assert path is not None
    assert path[-1] == (18, 18)


def test_astar_around_obstacle():
    grid, _ = room_with_obstacle(20, 20)
    path = astar_cells(grid, (2, 10), (15, 10))
    assert path is not None
    # Path must go around the wall gap at row 15
    for c, r in path:
        assert grid[r, c] == 0, f"Cell ({c},{r}) is inside an obstacle"


def test_astar_no_path():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 5] = 255   # solid wall
    result = astar_cells(grid, (2, 5), (8, 5))
    assert result is None


def test_astar_start_in_obstacle():
    grid, _ = open_room()
    grid[5, 5] = 255
    with pytest.raises(ValueError):
        astar_cells(grid, (5, 5), (10, 10))


# ── Douglas-Peucker ───────────────────────────────────────────────────────────

def test_dp_straight_line():
    pts = [(float(i), 0.0) for i in range(10)]
    simplified = douglas_peucker(pts, epsilon=0.01)
    assert len(simplified) == 2
    assert simplified[0] == pts[0]
    assert simplified[-1] == pts[-1]


def test_dp_corner():
    pts = [(0, 0), (5, 0), (5, 5)]
    simplified = douglas_peucker(pts, epsilon=0.01)
    assert len(simplified) == 3   # corner must be kept


def test_dp_preserves_endpoints():
    pts = [(0, 0), (1, 0.5), (2, 0), (3, 0.3), (4, 0)]
    simplified = douglas_peucker(pts, epsilon=0.01)
    assert simplified[0] == pts[0]
    assert simplified[-1] == pts[-1]


# ── enforce_min_spacing ───────────────────────────────────────────────────────

def test_min_spacing_basic():
    pts = [(float(i) * 0.1, 0.0) for i in range(20)]  # 0.1m apart
    spaced = enforce_min_spacing(pts, min_dist=0.5)
    for i in range(len(spaced) - 1):
        d = math.hypot(spaced[i+1][0] - spaced[i][0], spaced[i+1][1] - spaced[i][1])
        assert d >= 0.5 - 1e-9 or i == len(spaced) - 2  # last point is always kept


def test_min_spacing_preserves_endpoints():
    pts = [(float(i), 0.0) for i in range(5)]
    spaced = enforce_min_spacing(pts, min_dist=10.0)
    assert spaced[0] == pts[0]
    assert spaced[-1] == pts[-1]


# ── cells_to_waypoints ────────────────────────────────────────────────────────

def test_cells_to_waypoints_headings():
    meta = GridMeta(resolution=1.0, origin_x=0.0, origin_y=0.0, width=20, height=20)
    cell_path = [(i, 5) for i in range(1, 15)]   # horizontal path
    wps = cells_to_waypoints(cell_path, meta, dp_epsilon=0.1, min_spacing=0.5)
    assert len(wps) >= 2
    # All headings should be roughly 0° (east) for a horizontal path
    for wp in wps:
        assert abs(wp.theta) < 0.1, f"Unexpected heading {math.degrees(wp.theta):.1f}°"


def test_cells_to_waypoints_north():
    meta = GridMeta(resolution=1.0, origin_x=0.0, origin_y=0.0, width=20, height=20)
    cell_path = [(5, i) for i in range(1, 15)]   # vertical path (north = +Y)
    wps = cells_to_waypoints(cell_path, meta, dp_epsilon=0.1, min_spacing=0.5)
    # Heading should be ~90° (+Y)
    for wp in wps:
        assert abs(wp.theta - math.pi / 2) < 0.2


# ── plan_path (integration) ───────────────────────────────────────────────────

def test_plan_path_open_room():
    grid, meta = open_room(40, 40)
    # scale meta to 0.5m/cell
    meta = GridMeta(resolution=0.5, origin_x=0.0, origin_y=0.0, width=40, height=40)
    wps = plan_path(grid, (1.5, 1.5), (18.5, 18.5), meta)
    assert len(wps) >= 2
    assert wps[0].x == pytest.approx(1.5, abs=0.6)
    assert wps[-1].x == pytest.approx(18.5, abs=0.6)


def test_plan_path_no_path_raises():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 5] = 255
    meta = GridMeta(resolution=1.0, origin_x=0.0, origin_y=0.0, width=10, height=10)
    with pytest.raises(RuntimeError):
        plan_path(grid, (2.0, 5.0), (7.0, 5.0), meta)


# ── Parametric grid ───────────────────────────────────────────────────────────

def test_parametric_grid_shape():
    grid, meta = build_parametric_grid(resolution=0.1, arena_w=20.0, arena_h=15.0)
    assert grid.shape == (meta.height, meta.width)
    assert meta.width  == pytest.approx(200, abs=2)
    assert meta.height == pytest.approx(150, abs=2)


def test_parametric_grid_walls():
    grid, meta = build_parametric_grid(resolution=0.1)
    # Bottom row (south wall) should be occupied
    assert grid[0, meta.width // 2] == 255
    # Interior should have some free cells
    assert grid[meta.height // 2, meta.width // 2] == 0


def test_inflate_increases_occupied():
    grid, meta = build_parametric_grid(resolution=0.1)
    inflated = inflate_grid(grid, robot_radius_m=0.3, resolution=0.1)
    assert inflated.sum() > grid.sum()
    assert inflated.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
