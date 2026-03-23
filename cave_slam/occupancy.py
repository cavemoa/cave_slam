from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .slam import ScanMeasurement


@dataclass(frozen=True)
class OccupancyGridConfig:
    enabled: bool
    cell_size: float
    width: float
    height: float
    origin_x: float
    origin_y: float
    log_odds_hit: float
    log_odds_free: float
    log_odds_min: float
    log_odds_max: float
    occupied_threshold: float
    free_threshold: float
    ray_subsample: int


@dataclass
class OccupancyGridState:
    log_odds: np.ndarray
    width_cells: int
    height_cells: int
    cell_size: float
    origin_x: float
    origin_y: float


def initialize_occupancy_grid(config: OccupancyGridConfig):
    width_cells = max(1, int(np.ceil(config.width / config.cell_size)))
    height_cells = max(1, int(np.ceil(config.height / config.cell_size)))
    return OccupancyGridState(
        log_odds=np.zeros((height_cells, width_cells), dtype=float),
        width_cells=width_cells,
        height_cells=height_cells,
        cell_size=config.cell_size,
        origin_x=config.origin_x,
        origin_y=config.origin_y,
    )


def world_to_grid(x: float, y: float, grid_state: OccupancyGridState):
    col = int(np.floor((x - grid_state.origin_x) / grid_state.cell_size))
    row = int(np.floor((y - grid_state.origin_y) / grid_state.cell_size))
    return row, col


def grid_to_world(row: int, col: int, grid_state: OccupancyGridState):
    x = grid_state.origin_x + (col + 0.5) * grid_state.cell_size
    y = grid_state.origin_y + (row + 0.5) * grid_state.cell_size
    return x, y


def occupancy_probability(log_odds: float):
    return float(1.0 / (1.0 + np.exp(-log_odds)))


def _is_in_bounds(row: int, col: int, grid_state: OccupancyGridState):
    return 0 <= row < grid_state.height_cells and 0 <= col < grid_state.width_cells


def _clip_log_odds(
    current_value: float,
    delta: float,
    config: OccupancyGridConfig,
):
    return float(np.clip(current_value + delta, config.log_odds_min, config.log_odds_max))


def bresenham_cells(start_cell: tuple[int, int], end_cell: tuple[int, int]):
    row0, col0 = start_cell
    row1, col1 = end_cell

    d_row = abs(row1 - row0)
    d_col = abs(col1 - col0)
    step_row = 1 if row0 < row1 else -1
    step_col = 1 if col0 < col1 else -1

    row, col = row0, col0
    cells = [(row, col)]

    if d_col > d_row:
        error = d_col / 2
        while col != col1:
            col += step_col
            error -= d_row
            if error < 0:
                row += step_row
                error += d_col
            cells.append((row, col))
    else:
        error = d_row / 2
        while row != row1:
            row += step_row
            error -= d_col
            if error < 0:
                col += step_col
                error += d_row
            cells.append((row, col))

    return cells


def update_occupancy_from_scan(
    pose: np.ndarray,
    scan_samples: Sequence[ScanMeasurement],
    grid_state: OccupancyGridState,
    sensor_max_range: float,
    occupancy_config: OccupancyGridConfig,
):
    if not occupancy_config.enabled or not scan_samples:
        return grid_state

    pose = np.asarray(pose, dtype=float)
    origin_cell = world_to_grid(float(pose[0]), float(pose[1]), grid_state)
    if not _is_in_bounds(origin_cell[0], origin_cell[1], grid_state):
        return grid_state

    ray_step = max(1, int(occupancy_config.ray_subsample))
    hit_epsilon = 1e-9

    for measurement in scan_samples[::ray_step]:
        ray_theta = float(pose[2] + measurement.angle)
        end_x = float(pose[0] + measurement.distance * np.cos(ray_theta))
        end_y = float(pose[1] + measurement.distance * np.sin(ray_theta))
        end_cell = world_to_grid(end_x, end_y, grid_state)
        if not _is_in_bounds(end_cell[0], end_cell[1], grid_state):
            continue

        traversed_cells = bresenham_cells(origin_cell, end_cell)
        is_hit = float(measurement.distance) < (sensor_max_range - hit_epsilon)
        free_cells = traversed_cells[:-1] if is_hit else traversed_cells

        for row, col in free_cells:
            if _is_in_bounds(row, col, grid_state):
                grid_state.log_odds[row, col] = _clip_log_odds(
                    grid_state.log_odds[row, col],
                    -occupancy_config.log_odds_free,
                    occupancy_config,
                )

        if is_hit and traversed_cells:
            row, col = traversed_cells[-1]
            if _is_in_bounds(row, col, grid_state):
                grid_state.log_odds[row, col] = _clip_log_odds(
                    grid_state.log_odds[row, col],
                    occupancy_config.log_odds_hit,
                    occupancy_config,
                )

    return grid_state


def extract_occupied_cells(grid_state: OccupancyGridState, occupancy_config: OccupancyGridConfig):
    occupied_cells: list[tuple[float, float, float]] = []
    probabilities = 1.0 / (1.0 + np.exp(-grid_state.log_odds))
    occupied_indices = np.argwhere(probabilities >= occupancy_config.occupied_threshold)
    for row, col in occupied_indices:
        x, y = grid_to_world(int(row), int(col), grid_state)
        occupied_cells.append((x, y, float(probabilities[row, col])))
    return occupied_cells
