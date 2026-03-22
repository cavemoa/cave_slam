from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import yaml

from .agent import AgentState, MotionCommand, initialize_agent_state, step_agent
from .slam import (
    EkfAssociationSummary,
    EkfDebugInfo,
    EkfStepDiagnostics,
    LidarScan,
    EkfUpdateResult,
    PersistentLandmark,
    ScanMeasurement,
    TruthObservationSet,
    VoxelCellState,
    WallSegment,
    build_ekf_step_diagnostics,
    compute_ekf_debug_info,
    ekf_predict,
    ekf_update_pose_only_batch,
    extract_landmarks,
    extract_truth_landmark_positions,
    measurements_to_world_points,
    simulate_lidar,
    simulate_landmark_observations_from_truth,
    transform_measurements,
    update_persistent_landmarks,
    update_voxel_grid,
)

if TYPE_CHECKING:
    from .viz import PlotArtists

DEFAULT_CONFIG_PATH = "cave_slam.yaml"
DEFAULT_CONFIG = {
    "simulation": {
        "delay_ms": 50,
        "frames": 800,
        "random_seed": None,
    },
    "environment": {
        "generator": {
            "enabled": True,
            "complexity": 15,
            "width": 15.0,
            "height": 15.0,
            "overlap_boundaries": True,
            "min_area": 1.0,
            "max_area": 6.0,
        },
        "walls": [],
    },
    "sensor": {
        "fov_degrees": 60,
        "num_rays": 16,
        "max_range": 15.0,
        "noise": {
            "enabled": True,
            "min_relative_std": 0.005,
            "max_relative_std": 0.05,
        },
    },
    "feature_extraction": {
        "window_size": 2,
        "max_neighbor_gap": 1.0,
        "min_segment_span": 0.2,
        "max_line_residual": 0.12,
        "min_corner_angle_deg": 45,
        "max_corner_angle_deg": 135,
        "nms_radius": 1,
        "persistence_frames": 25,
        "association_radius": 0.35,
    },
    "motion": {
        "linear_speed": 0.2,
        "random_turn_deg": 5,
        "collision_turn_deg": 15,
        "collision_distance": 1.0,
        "clear_distance": 1.4,
        "clearance_frames": 3,
        "front_sector_deg": 10,
        "side_balance_tolerance": 0.1,
    },
    "odometry_noise": {
        "distance_std": 0.01,
        "angle_std_deg": 1.5,
    },
    "voxel_grid": {
        "voxel_size": 0.35,
        "min_points_per_voxel": 3,
        "point_size": 30,
        "point_alpha": 0.9,
        "color": "darkorange",
        "weighting_mode": "inverse_variance",
        "distance_weight_power": 2.0,
        "temporal_decay": 0.995,
        "best_observation_override": True,
        "best_observation_max_distance": 4.0,
    },
    "agent": {
        "initial_pose": {
            "x": 2.0,
            "y": 2.0,
            "theta_deg": 45,
        },
        "startup_behavior": {
            "enabled": True,
            "mode": "spin_in_place",
            "rotation_degrees": 360,
            "turn_step_deg": 15,
        },
    },
    "plot": {
        "figsize": [8, 8],
        "xlim": [-1, 16],
        "ylim": [-1, 16],
        "point_size": 10,
        "point_alpha": 0.08,
    },
    "ekf": {
        "measurement": {
            "model_type": "range_bearing",
            "range_std": 0.05,
            "bearing_std_deg": 2.0,
        },
        "truth_update": {
            "enabled": False,
            "max_observations": 8,
            "max_range": 12.0,
        },
        "pose_update": {
            "enabled": False,
            "use_truth_observations": True,
            "max_updates_per_frame": 8,
        },
    },
}


@dataclass(frozen=True)
class SimulationConfig:
    delay_ms: int
    frames: int
    random_seed: int | None


@dataclass(frozen=True)
class GeneratorConfig:
    enabled: bool
    complexity: int
    width: float
    height: float
    overlap_boundaries: bool
    min_area: float
    max_area: float


@dataclass(frozen=True)
class EnvironmentConfig:
    generator: GeneratorConfig
    walls: list[WallSegment]


@dataclass(frozen=True)
class SensorNoiseConfig:
    enabled: bool
    min_relative_std: float
    max_relative_std: float


@dataclass(frozen=True)
class SensorConfig:
    fov_degrees: float
    num_rays: int
    max_range: float
    noise: SensorNoiseConfig


@dataclass(frozen=True)
class FeatureExtractionConfig:
    window_size: int
    max_neighbor_gap: float
    min_segment_span: float
    max_line_residual: float
    min_corner_angle_deg: float
    max_corner_angle_deg: float
    nms_radius: int
    persistence_frames: int
    association_radius: float


@dataclass(frozen=True)
class MotionConfig:
    linear_speed: float
    random_turn_deg: float
    collision_turn_deg: float
    collision_distance: float
    clear_distance: float
    clearance_frames: int
    front_sector_deg: float
    side_balance_tolerance: float


@dataclass(frozen=True)
class OdometryNoiseConfig:
    distance_std: float
    angle_std_deg: float


@dataclass(frozen=True)
class VoxelGridConfig:
    voxel_size: float
    min_points_per_voxel: int
    point_size: float
    point_alpha: float
    color: str
    weighting_mode: str
    distance_weight_power: float
    temporal_decay: float
    best_observation_override: bool
    best_observation_max_distance: float


@dataclass(frozen=True)
class InitialPoseConfig:
    x: float
    y: float
    theta_deg: float


@dataclass(frozen=True)
class StartupBehaviorConfig:
    enabled: bool
    mode: str
    rotation_degrees: float
    turn_step_deg: float


@dataclass(frozen=True)
class AgentConfig:
    initial_pose: InitialPoseConfig
    startup_behavior: StartupBehaviorConfig


@dataclass(frozen=True)
class PlotConfig:
    figsize: tuple[float, float]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    point_size: float
    point_alpha: float


@dataclass(frozen=True)
class MeasurementModelConfig:
    model_type: str
    range_std: float
    bearing_std_deg: float


@dataclass(frozen=True)
class TruthUpdateConfig:
    enabled: bool
    max_observations: int
    max_range: float


@dataclass(frozen=True)
class PoseUpdateConfig:
    enabled: bool
    use_truth_observations: bool
    max_updates_per_frame: int


@dataclass(frozen=True)
class EkfConfig:
    measurement: MeasurementModelConfig
    truth_update: TruthUpdateConfig
    pose_update: PoseUpdateConfig


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    environment: EnvironmentConfig
    sensor: SensorConfig
    feature_extraction: FeatureExtractionConfig
    motion: MotionConfig
    odometry_noise: OdometryNoiseConfig
    voxel_grid: VoxelGridConfig
    agent: AgentConfig
    plot: PlotConfig
    ekf: EkfConfig


@dataclass
class SlamState:
    mu: np.ndarray
    Sigma: np.ndarray
    point_cloud_x: list[float]
    point_cloud_y: list[float]
    voxel_points_x: list[float]
    voxel_points_y: list[float]
    voxel_state: dict
    persistent_landmarks: list[PersistentLandmark]
    true_trajectory_x: list[float]
    true_trajectory_y: list[float]
    ekf_trajectory_x: list[float]
    ekf_trajectory_y: list[float]


@dataclass
class StepResult:
    frame_index: int
    observation_pose: np.ndarray
    lidar_scan: LidarScan
    observed_landmarks: list[ScanMeasurement]
    truth_observation_set: TruthObservationSet | None
    association_result: EkfAssociationSummary | None
    motion_command: MotionCommand
    measured_distance: float
    measured_turn: float
    pose_update_results: list[EkfUpdateResult]
    ekf_diagnostics: EkfStepDiagnostics
    ekf_debug_info: EkfDebugInfo


@dataclass
class SimulationState:
    config: AppConfig
    rng: np.random.Generator
    walls: list[WallSegment]
    truth_landmark_positions: list[np.ndarray]
    agent_state: AgentState
    slam_state: SlamState
    step_index: int = 0
    is_paused: bool = False
    last_step_result: StepResult | None = None
    artists: PlotArtists | None = None
    animation: object | None = None


def deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_mapping(value: Any, path: str):
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _require_bool(value: Any, path: str):
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _require_int(value: Any, path: str):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _require_float(value: Any, path: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    return float(value)


def _require_optional_int(value: Any, path: str):
    if value is None:
        return None
    return _require_int(value, path)


def _require_str(value: Any, path: str):
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _require_choice(value: Any, path: str, choices: Sequence[str]):
    string_value = _require_str(value, path)
    if string_value not in choices:
        raise ValueError(f"{path} must be one of {', '.join(choices)}")
    return string_value


def _require_probability(value: Any, path: str):
    probability = _require_float(value, path)
    if not (0.0 < probability <= 1.0):
        raise ValueError(f"{path} must be in the interval (0, 1]")
    return probability


def _require_pair(value: Any, path: str):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{path} must be a two-item sequence")
    return (
        _require_float(value[0], f"{path}[0]"),
        _require_float(value[1], f"{path}[1]"),
    )


def _parse_wall(value: Any, path: str):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise TypeError(f"{path} must contain exactly two points")
    return WallSegment(
        start=_require_pair(value[0], f"{path}[0]"),
        end=_require_pair(value[1], f"{path}[1]"),
    )


def parse_config(raw_config: Mapping[str, Any]):
    raw = _require_mapping(raw_config, "config")

    simulation = _require_mapping(raw["simulation"], "simulation")
    environment = _require_mapping(raw["environment"], "environment")
    generator = _require_mapping(environment["generator"], "environment.generator")
    sensor = _require_mapping(raw["sensor"], "sensor")
    sensor_noise = _require_mapping(sensor["noise"], "sensor.noise")
    feature_extraction = _require_mapping(raw["feature_extraction"], "feature_extraction")
    motion = _require_mapping(raw["motion"], "motion")
    odometry_noise = _require_mapping(raw["odometry_noise"], "odometry_noise")
    voxel_grid = _require_mapping(raw["voxel_grid"], "voxel_grid")
    agent = _require_mapping(raw["agent"], "agent")
    initial_pose = _require_mapping(agent["initial_pose"], "agent.initial_pose")
    startup_behavior = _require_mapping(agent["startup_behavior"], "agent.startup_behavior")
    plot = _require_mapping(raw["plot"], "plot")
    ekf = _require_mapping(raw["ekf"], "ekf")
    ekf_measurement = _require_mapping(ekf["measurement"], "ekf.measurement")
    ekf_truth_update = _require_mapping(ekf["truth_update"], "ekf.truth_update")
    ekf_pose_update = _require_mapping(ekf["pose_update"], "ekf.pose_update")

    walls_raw = environment.get("walls", [])
    if not isinstance(walls_raw, Sequence) or isinstance(walls_raw, (str, bytes)):
        raise TypeError("environment.walls must be a sequence")

    return AppConfig(
        simulation=SimulationConfig(
            delay_ms=_require_int(simulation["delay_ms"], "simulation.delay_ms"),
            frames=_require_int(simulation["frames"], "simulation.frames"),
            random_seed=_require_optional_int(simulation["random_seed"], "simulation.random_seed"),
        ),
        environment=EnvironmentConfig(
            generator=GeneratorConfig(
                enabled=_require_bool(generator["enabled"], "environment.generator.enabled"),
                complexity=_require_int(generator["complexity"], "environment.generator.complexity"),
                width=_require_float(generator["width"], "environment.generator.width"),
                height=_require_float(generator["height"], "environment.generator.height"),
                overlap_boundaries=_require_bool(generator["overlap_boundaries"], "environment.generator.overlap_boundaries"),
                min_area=_require_float(generator["min_area"], "environment.generator.min_area"),
                max_area=_require_float(generator["max_area"], "environment.generator.max_area"),
            ),
            walls=[_parse_wall(wall, f"environment.walls[{index}]") for index, wall in enumerate(walls_raw)],
        ),
        sensor=SensorConfig(
            fov_degrees=_require_float(sensor["fov_degrees"], "sensor.fov_degrees"),
            num_rays=_require_int(sensor["num_rays"], "sensor.num_rays"),
            max_range=_require_float(sensor["max_range"], "sensor.max_range"),
            noise=SensorNoiseConfig(
                enabled=_require_bool(sensor_noise["enabled"], "sensor.noise.enabled"),
                min_relative_std=_require_float(sensor_noise["min_relative_std"], "sensor.noise.min_relative_std"),
                max_relative_std=_require_float(sensor_noise["max_relative_std"], "sensor.noise.max_relative_std"),
            ),
        ),
        feature_extraction=FeatureExtractionConfig(
            window_size=_require_int(feature_extraction["window_size"], "feature_extraction.window_size"),
            max_neighbor_gap=_require_float(feature_extraction["max_neighbor_gap"], "feature_extraction.max_neighbor_gap"),
            min_segment_span=_require_float(feature_extraction["min_segment_span"], "feature_extraction.min_segment_span"),
            max_line_residual=_require_float(feature_extraction["max_line_residual"], "feature_extraction.max_line_residual"),
            min_corner_angle_deg=_require_float(feature_extraction["min_corner_angle_deg"], "feature_extraction.min_corner_angle_deg"),
            max_corner_angle_deg=_require_float(feature_extraction["max_corner_angle_deg"], "feature_extraction.max_corner_angle_deg"),
            nms_radius=_require_int(feature_extraction["nms_radius"], "feature_extraction.nms_radius"),
            persistence_frames=_require_int(feature_extraction["persistence_frames"], "feature_extraction.persistence_frames"),
            association_radius=_require_float(feature_extraction["association_radius"], "feature_extraction.association_radius"),
        ),
        motion=MotionConfig(
            linear_speed=_require_float(motion["linear_speed"], "motion.linear_speed"),
            random_turn_deg=_require_float(motion["random_turn_deg"], "motion.random_turn_deg"),
            collision_turn_deg=_require_float(motion["collision_turn_deg"], "motion.collision_turn_deg"),
            collision_distance=_require_float(motion["collision_distance"], "motion.collision_distance"),
            clear_distance=_require_float(motion["clear_distance"], "motion.clear_distance"),
            clearance_frames=_require_int(motion["clearance_frames"], "motion.clearance_frames"),
            front_sector_deg=_require_float(motion["front_sector_deg"], "motion.front_sector_deg"),
            side_balance_tolerance=_require_float(motion["side_balance_tolerance"], "motion.side_balance_tolerance"),
        ),
        odometry_noise=OdometryNoiseConfig(
            distance_std=_require_float(odometry_noise["distance_std"], "odometry_noise.distance_std"),
            angle_std_deg=_require_float(odometry_noise["angle_std_deg"], "odometry_noise.angle_std_deg"),
        ),
        voxel_grid=VoxelGridConfig(
            voxel_size=_require_float(voxel_grid["voxel_size"], "voxel_grid.voxel_size"),
            min_points_per_voxel=_require_int(voxel_grid["min_points_per_voxel"], "voxel_grid.min_points_per_voxel"),
            point_size=_require_float(voxel_grid["point_size"], "voxel_grid.point_size"),
            point_alpha=_require_float(voxel_grid["point_alpha"], "voxel_grid.point_alpha"),
            color=_require_str(voxel_grid["color"], "voxel_grid.color"),
            weighting_mode=_require_choice(voxel_grid["weighting_mode"], "voxel_grid.weighting_mode", ("inverse_variance", "inverse_distance")),
            distance_weight_power=_require_float(voxel_grid["distance_weight_power"], "voxel_grid.distance_weight_power"),
            temporal_decay=_require_probability(voxel_grid["temporal_decay"], "voxel_grid.temporal_decay"),
            best_observation_override=_require_bool(voxel_grid["best_observation_override"], "voxel_grid.best_observation_override"),
            best_observation_max_distance=_require_float(voxel_grid["best_observation_max_distance"], "voxel_grid.best_observation_max_distance"),
        ),
        agent=AgentConfig(
            initial_pose=InitialPoseConfig(
                x=_require_float(initial_pose["x"], "agent.initial_pose.x"),
                y=_require_float(initial_pose["y"], "agent.initial_pose.y"),
                theta_deg=_require_float(initial_pose["theta_deg"], "agent.initial_pose.theta_deg"),
            ),
            startup_behavior=StartupBehaviorConfig(
                enabled=_require_bool(startup_behavior["enabled"], "agent.startup_behavior.enabled"),
                mode=_require_str(startup_behavior["mode"], "agent.startup_behavior.mode"),
                rotation_degrees=_require_float(startup_behavior["rotation_degrees"], "agent.startup_behavior.rotation_degrees"),
                turn_step_deg=_require_float(startup_behavior["turn_step_deg"], "agent.startup_behavior.turn_step_deg"),
            ),
        ),
        plot=PlotConfig(
            figsize=_require_pair(plot["figsize"], "plot.figsize"),
            xlim=_require_pair(plot["xlim"], "plot.xlim"),
            ylim=_require_pair(plot["ylim"], "plot.ylim"),
            point_size=_require_float(plot["point_size"], "plot.point_size"),
            point_alpha=_require_float(plot["point_alpha"], "plot.point_alpha"),
        ),
        ekf=EkfConfig(
            measurement=MeasurementModelConfig(
                model_type=_require_choice(ekf_measurement["model_type"], "ekf.measurement.model_type", ("range_bearing",)),
                range_std=_require_float(ekf_measurement["range_std"], "ekf.measurement.range_std"),
                bearing_std_deg=_require_float(ekf_measurement["bearing_std_deg"], "ekf.measurement.bearing_std_deg"),
            ),
            truth_update=TruthUpdateConfig(
                enabled=_require_bool(ekf_truth_update["enabled"], "ekf.truth_update.enabled"),
                max_observations=_require_int(ekf_truth_update["max_observations"], "ekf.truth_update.max_observations"),
                max_range=_require_float(ekf_truth_update["max_range"], "ekf.truth_update.max_range"),
            ),
            pose_update=PoseUpdateConfig(
                enabled=_require_bool(ekf_pose_update["enabled"], "ekf.pose_update.enabled"),
                use_truth_observations=_require_bool(ekf_pose_update["use_truth_observations"], "ekf.pose_update.use_truth_observations"),
                max_updates_per_frame=_require_int(ekf_pose_update["max_updates_per_frame"], "ekf.pose_update.max_updates_per_frame"),
            ),
        ),
    )


def build_config(overrides: Mapping[str, Any] | None = None):
    merged = deep_merge(DEFAULT_CONFIG, overrides or {})
    return parse_config(merged)


def load_config(path: str):
    if not os.path.exists(path):
        print(f"Config file {path} not found. Using defaults.")
        return build_config()

    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return build_config(loaded)


def generate_environment(
    seed: int | None = None,
    complexity: int = 5,
    bounds: tuple[float, float, float, float] = (0, 0, 15, 15),
    safe_zone: tuple[float, float, float] = (2.0, 2.0, 2.0),
    overlap_boundaries: bool = True,
    min_area: float = 1.0,
    max_area: float = 6.0,
    max_attempts_per_obstacle: int = 1000,
):
    rng = np.random.default_rng(seed)
    walls: list[WallSegment] = []

    min_x, min_y, max_x, max_y = bounds
    walls.extend(
        [
            WallSegment((min_x, min_y), (max_x, min_y)),
            WallSegment((max_x, min_y), (max_x, max_y)),
            WallSegment((max_x, max_y), (min_x, max_y)),
            WallSegment((min_x, max_y), (min_x, min_y)),
        ]
    )

    for obstacle_index in range(complexity):
        target_area = rng.uniform(min_area, max_area)
        num_sides = rng.integers(3, 7)

        angles = np.sort(rng.uniform(0, 2 * np.pi, num_sides))
        base_radii = rng.uniform(0.4, 1.0, num_sides)

        base_area = 0.0
        for i in range(num_sides):
            d_theta = angles[(i + 1) % num_sides] - angles[i]
            if d_theta < 0:
                d_theta += 2 * np.pi
            base_area += 0.5 * base_radii[i] * base_radii[(i + 1) % num_sides] * np.sin(d_theta)

        scale_factor = np.sqrt(target_area / base_area)
        final_radii = base_radii * scale_factor
        max_radius = float(np.max(final_radii))

        valid_position = False
        for _ in range(max_attempts_per_obstacle):
            if overlap_boundaries:
                cx = rng.uniform(min_x, max_x)
                cy = rng.uniform(min_y, max_y)
            else:
                cx = rng.uniform(min_x + max_radius, max_x - max_radius)
                cy = rng.uniform(min_y + max_radius, max_y - max_radius)

            dist = np.sqrt((cx - safe_zone[0]) ** 2 + (cy - safe_zone[1]) ** 2)
            if dist > (safe_zone[2] + max_radius):
                valid_position = True
                break

        if not valid_position:
            raise RuntimeError(
                f"Could not place obstacle {obstacle_index} within {max_attempts_per_obstacle} attempts. "
                "Relax generator constraints or increase the environment size."
            )

        polygon_corners = []
        for angle, radius in zip(angles, final_radii):
            px = cx + radius * np.cos(angle)
            py = cy + radius * np.sin(angle)
            polygon_corners.append((float(px), float(py)))

        for i in range(num_sides):
            p1 = polygon_corners[i]
            p2 = polygon_corners[(i + 1) % num_sides]
            walls.append(WallSegment(p1, p2))

    return walls


def initialize_slam_state(initial_pose: InitialPoseConfig):
    mu = np.array(
        [initial_pose.x, initial_pose.y, np.radians(initial_pose.theta_deg)],
        dtype=float,
    )
    sigma = np.zeros((3, 3), dtype=float)

    return SlamState(
        mu=mu,
        Sigma=sigma,
        point_cloud_x=[],
        point_cloud_y=[],
        voxel_points_x=[],
        voxel_points_y=[],
        voxel_state=defaultdict(VoxelCellState),
        persistent_landmarks=[],
        true_trajectory_x=[initial_pose.x],
        true_trajectory_y=[initial_pose.y],
        ekf_trajectory_x=[initial_pose.x],
        ekf_trajectory_y=[initial_pose.y],
    )


def create_simulation(config: AppConfig | Mapping[str, Any]):
    if not isinstance(config, AppConfig):
        config = build_config(config)

    rng = np.random.default_rng(config.simulation.random_seed)

    if config.environment.generator.enabled:
        generator = config.environment.generator
        walls = generate_environment(
            seed=config.simulation.random_seed,
            complexity=generator.complexity,
            bounds=(0.0, 0.0, generator.width, generator.height),
            safe_zone=(config.agent.initial_pose.x, config.agent.initial_pose.y, 2.0),
            overlap_boundaries=generator.overlap_boundaries,
            min_area=generator.min_area,
            max_area=generator.max_area,
        )
    else:
        walls = list(config.environment.walls)

    return SimulationState(
        config=config,
        rng=rng,
        walls=walls,
        truth_landmark_positions=extract_truth_landmark_positions(walls),
        agent_state=initialize_agent_state(config.agent.initial_pose, config.agent.startup_behavior),
        slam_state=initialize_slam_state(config.agent.initial_pose),
    )


def extract_truth_landmarks_for_update(state: SimulationState):
    return state.truth_landmark_positions


def build_truth_observations(state: SimulationState, observation_pose: np.ndarray):
    truth_update_config = state.config.ekf.truth_update
    if not truth_update_config.enabled:
        return None

    return simulate_landmark_observations_from_truth(
        observation_pose,
        extract_truth_landmarks_for_update(state),
        state.config.ekf.measurement,
        state.config.sensor,
        state.rng,
        max_range=min(truth_update_config.max_range, state.config.sensor.max_range),
        max_observations=truth_update_config.max_observations,
    )


def apply_pose_only_ekf_correction(state: SimulationState, truth_observation_set: TruthObservationSet | None):
    pose_update_config = state.config.ekf.pose_update
    if not pose_update_config.enabled or not pose_update_config.use_truth_observations or truth_observation_set is None:
        return []

    max_updates = max(0, pose_update_config.max_updates_per_frame)
    if max_updates == 0:
        return []

    selected_observations = truth_observation_set.observations[:max_updates]
    selected_landmarks = truth_observation_set.landmark_positions[:max_updates]
    if not selected_observations:
        return []

    updated_mu, updated_Sigma, update_results = ekf_update_pose_only_batch(
        state.slam_state.mu,
        state.slam_state.Sigma,
        selected_observations,
        selected_landmarks,
        state.config.ekf.measurement,
    )
    state.slam_state.mu = updated_mu
    state.slam_state.Sigma = updated_Sigma
    return update_results


def step_simulation(state: SimulationState):
    current_frame = state.step_index + 1
    observation_pose = state.agent_state.true_pose.copy()
    truth_observation_set = build_truth_observations(state, observation_pose)
    forward_sector_rad = np.radians(state.config.motion.front_sector_deg)
    lidar_scan = simulate_lidar(
        observation_pose[0],
        observation_pose[1],
        observation_pose[2],
        state.walls,
        state.config.sensor,
        state.rng,
        forward_sector_rad,
    )

    slam_state = state.slam_state
    observed_landmarks = extract_landmarks(lidar_scan.measurements, state.config.feature_extraction)
    observed_landmark_points = measurements_to_world_points(observed_landmarks, observation_pose)
    slam_state.persistent_landmarks = update_persistent_landmarks(
        observed_landmark_points,
        slam_state.persistent_landmarks,
        state.config.feature_extraction,
    )

    mapped_x, mapped_y = transform_measurements(lidar_scan.measurements, slam_state.mu)
    slam_state.point_cloud_x.extend(mapped_x)
    slam_state.point_cloud_y.extend(mapped_y)

    averaged_x, averaged_y = update_voxel_grid(
        mapped_x,
        mapped_y,
        lidar_scan.measurements,
        slam_state.voxel_state,
        state.config.voxel_grid,
        state.config.sensor,
        current_frame,
    )
    slam_state.voxel_points_x[:] = averaged_x
    slam_state.voxel_points_y[:] = averaged_y

    command = step_agent(
        state.agent_state,
        lidar_scan.min_dist_forward,
        lidar_scan.scan_samples,
        state.config.motion,
        state.config.agent.startup_behavior,
        state.walls,
        state.rng,
    )

    measured_turn = command.turn + np.radians(state.config.odometry_noise.angle_std_deg) * state.rng.normal(0.0, 1.0)
    if command.distance == 0.0:
        measured_distance = 0.0
    else:
        measured_distance = max(0.0, command.distance + state.rng.normal(0.0, state.config.odometry_noise.distance_std))

    slam_state.mu, slam_state.Sigma = ekf_predict(
        slam_state.mu,
        slam_state.Sigma,
        measured_distance,
        measured_turn,
        state.config.odometry_noise,
    )
    pre_update_mu = slam_state.mu.copy()
    pre_update_sigma = slam_state.Sigma.copy()
    pose_update_results = apply_pose_only_ekf_correction(state, truth_observation_set)
    num_candidate_observations = len(truth_observation_set.observations) if truth_observation_set is not None else 0
    association_result = None
    if truth_observation_set is not None and state.config.ekf.pose_update.use_truth_observations:
        association_result = EkfAssociationSummary(
            num_candidate_observations=num_candidate_observations,
            num_matches=len(pose_update_results),
            num_rejections=0,
            source="truth_harness",
            gating_applied=False,
        )
    ekf_diagnostics = build_ekf_step_diagnostics(
        pose_before_update=pre_update_mu,
        pose_after_update=slam_state.mu,
        sigma_before_update=pre_update_sigma,
        sigma_after_update=slam_state.Sigma,
        update_results=pose_update_results,
        num_candidate_observations=num_candidate_observations,
        num_rejections=0,
    )
    ekf_debug_info = compute_ekf_debug_info(slam_state.Sigma)

    slam_state.true_trajectory_x.append(state.agent_state.true_pose[0])
    slam_state.true_trajectory_y.append(state.agent_state.true_pose[1])
    slam_state.ekf_trajectory_x.append(slam_state.mu[0])
    slam_state.ekf_trajectory_y.append(slam_state.mu[1])
    state.step_index = current_frame

    result = StepResult(
        frame_index=current_frame,
        observation_pose=observation_pose,
        lidar_scan=lidar_scan,
        observed_landmarks=observed_landmarks,
        truth_observation_set=truth_observation_set,
        association_result=association_result,
        motion_command=command,
        measured_distance=measured_distance,
        measured_turn=measured_turn,
        pose_update_results=pose_update_results,
        ekf_diagnostics=ekf_diagnostics,
        ekf_debug_info=ekf_debug_info,
    )
    state.last_step_result = result
    return result
