"""Shared datatypes for the VLA4AMR hybrid nav stack."""
from dataclasses import dataclass, field
from typing import NamedTuple
import numpy as np


@dataclass
class GridMeta:
    """Metadata for converting between grid cells and world (metric) coordinates."""
    resolution: float       # metres per cell
    origin_x: float         # world X of cell (0, 0) — metres
    origin_y: float         # world Y of cell (0, 0) — metres
    width: int              # grid columns
    height: int             # grid rows

    def world_to_cell(self, wx: float, wy: float) -> tuple[int, int]:
        """World (x, y) → (col, row). Raises ValueError if outside grid."""
        col = int((wx - self.origin_x) / self.resolution)
        row = int((wy - self.origin_y) / self.resolution)
        if not (0 <= col < self.width and 0 <= row < self.height):
            raise ValueError(
                f"World ({wx:.2f}, {wy:.2f}) maps to cell ({col}, {row}) "
                f"which is outside grid {self.width}×{self.height}"
            )
        return col, row

    def cell_to_world(self, col: int, row: int) -> tuple[float, float]:
        """(col, row) → world (x, y) at cell centre."""
        wx = self.origin_x + (col + 0.5) * self.resolution
        wy = self.origin_y + (row + 0.5) * self.resolution
        return wx, wy


class Waypoint(NamedTuple):
    x: float       # world metres
    y: float       # world metres
    theta: float   # heading in radians, 0 = +X axis
