from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .slam import ScanMeasurement, WallSegment, get_intersection

if TYPE_CHECKING:
    from .sim import MotionConfig, StartupBehaviorConfig


@dataclass
class MotionCommand:
    turn: float
    distance: float


@dataclass
class AgentState:
    true_pose: np.ndarray
    startup_rotation_remaining: float = 0.0
    avoiding_obstacle: bool = False
    avoidance_turn_sign: float = 1.0
    avoidance_clear_count: int = 0


def initialize_agent_state(initial_pose, startup_config: StartupBehaviorConfig):
    startup_rotation_remaining = 0.0
    if startup_config.enabled:
        startup_rotation_remaining = np.radians(max(0.0, startup_config.rotation_degrees))

    true_pose = np.array(
        [initial_pose.x, initial_pose.y, np.radians(initial_pose.theta_deg)],
        dtype=float,
    )
    return AgentState(true_pose=true_pose, startup_rotation_remaining=startup_rotation_remaining)


def is_path_clear(x: float, y: float, next_x: float, next_y: float, walls: Sequence[WallSegment], buffer: float = 0.15):
    dx = next_x - x
    dy = next_y - y
    move_dist = np.hypot(dx, dy)

    if move_dist < 1e-6:
        return True

    ray_dir = [dx / move_dist, dy / move_dist]

    for wall in walls:
        hit, dist = get_intersection((x, y), ray_dir, wall)
        if hit is not None and dist < (move_dist + buffer):
            return False

    return True


def choose_avoidance_turn_sign(scan_samples: Sequence[ScanMeasurement], motion_config: MotionConfig, rng):
    front_sector_rad = np.radians(motion_config.front_sector_deg)
    side_balance_tolerance = motion_config.side_balance_tolerance

    left_samples = [measurement for measurement in scan_samples if measurement.angle > front_sector_rad]
    right_samples = [measurement for measurement in scan_samples if measurement.angle < -front_sector_rad]

    def weighted_clearance(samples: Sequence[ScanMeasurement]):
        if not samples:
            return 0.0

        angles = np.array([measurement.angle for measurement in samples], dtype=float)
        distances = np.array([measurement.distance for measurement in samples], dtype=float)
        weights = np.cos(angles) ** 2
        return float(np.average(distances, weights=weights))

    left_score = weighted_clearance(left_samples)
    right_score = weighted_clearance(right_samples)

    if abs(left_score - right_score) <= side_balance_tolerance:
        return float(rng.choice((-1.0, 1.0)))
    return 1.0 if left_score > right_score else -1.0


def compute_motion_command(
    agent_state: AgentState,
    dist_ahead: float,
    scan_samples: Sequence[ScanMeasurement],
    motion_config: MotionConfig,
    startup_config: StartupBehaviorConfig,
    rng,
):
    collision_turn = np.radians(motion_config.collision_turn_deg)
    obstacle_ahead = dist_ahead < motion_config.collision_distance

    if agent_state.startup_rotation_remaining > 0:
        commanded_turn = min(np.radians(startup_config.turn_step_deg), agent_state.startup_rotation_remaining)
        agent_state.startup_rotation_remaining -= commanded_turn
        return MotionCommand(turn=commanded_turn, distance=0.0)

    if obstacle_ahead and not agent_state.avoiding_obstacle:
        agent_state.avoidance_turn_sign = choose_avoidance_turn_sign(scan_samples, motion_config, rng)
        agent_state.avoiding_obstacle = True
        agent_state.avoidance_clear_count = 0

    if agent_state.avoiding_obstacle:
        if dist_ahead > motion_config.clear_distance:
            agent_state.avoidance_clear_count += 1
            if agent_state.avoidance_clear_count >= motion_config.clearance_frames:
                agent_state.avoiding_obstacle = False
                agent_state.avoidance_clear_count = 0
        else:
            agent_state.avoidance_clear_count = 0

    if agent_state.avoiding_obstacle:
        return MotionCommand(
            turn=agent_state.avoidance_turn_sign * collision_turn,
            distance=0.0,
        )

    return MotionCommand(
        turn=np.radians(rng.uniform(-motion_config.random_turn_deg, motion_config.random_turn_deg)),
        distance=motion_config.linear_speed,
    )


def apply_bumper(
    agent_state: AgentState,
    command: MotionCommand,
    walls: Sequence[WallSegment],
    scan_samples: Sequence[ScanMeasurement],
    motion_config: MotionConfig,
    rng,
):
    proposed_theta = agent_state.true_pose[2] + command.turn
    proposed_x = agent_state.true_pose[0] + command.distance * np.cos(proposed_theta)
    proposed_y = agent_state.true_pose[1] + command.distance * np.sin(proposed_theta)

    if is_path_clear(agent_state.true_pose[0], agent_state.true_pose[1], proposed_x, proposed_y, walls):
        return command

    if not agent_state.avoiding_obstacle:
        agent_state.avoidance_turn_sign = choose_avoidance_turn_sign(scan_samples, motion_config, rng)
        agent_state.avoiding_obstacle = True

    agent_state.avoidance_clear_count = 0
    collision_turn = np.radians(motion_config.collision_turn_deg)
    return MotionCommand(turn=agent_state.avoidance_turn_sign * collision_turn, distance=0.0)


def apply_motion_command(agent_state: AgentState, command: MotionCommand):
    agent_state.true_pose[2] += command.turn
    agent_state.true_pose[0] += command.distance * np.cos(agent_state.true_pose[2])
    agent_state.true_pose[1] += command.distance * np.sin(agent_state.true_pose[2])


def step_agent(
    agent_state: AgentState,
    dist_ahead: float,
    scan_samples: Sequence[ScanMeasurement],
    motion_config: MotionConfig,
    startup_config: StartupBehaviorConfig,
    walls: Sequence[WallSegment],
    rng,
):
    command = compute_motion_command(agent_state, dist_ahead, scan_samples, motion_config, startup_config, rng)
    command = apply_bumper(agent_state, command, walls, scan_samples, motion_config, rng)
    apply_motion_command(agent_state, command)
    return command
