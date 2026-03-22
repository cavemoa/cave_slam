import argparse
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import yaml

def configure_matplotlib_backend():
    """Prefer a GUI backend that matches the active Linux display server."""
    if sys.platform.startswith("linux"):
        if "QT_QPA_PLATFORM" not in os.environ and os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"

        has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        matplotlib.use("qtagg" if has_display else "Agg")
        return

    matplotlib.use("qtagg")

configure_matplotlib_backend()
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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
        "walls": [], # Used if generator is disabled
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
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the cave SLAM simulation with EKF Prediction and Voxel Grids."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()

def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged

def load_config(path):
    if not os.path.exists(path):
        print(f"Config file {path} not found. Using defaults.")
        return DEFAULT_CONFIG
        
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return deep_merge(DEFAULT_CONFIG, loaded)

def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]

def get_intersection(ray_origin, ray_dir, wall):
    p = np.array(ray_origin)
    d = np.array(ray_dir)
    a = np.array(wall[0])
    b = np.array(wall[1])

    v1 = p - a
    v2 = b - a
    v3 = np.array([-d[1], d[0]])

    dot = np.dot(v2, v3)
    if abs(dot) < 1e-6:
        return None, float("inf")

    t1 = cross2d(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot

    if t1 >= 0 and 0 <= t2 <= 1:
        hit_point = p + t1 * d
        return hit_point, t1

    return None, float("inf")

def is_path_clear(x, y, next_x, next_y, walls, buffer=0.15):
    """Acts as a rigid physics bumper. Checks if the proposed movement line intersects any wall."""
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

def generate_environment(seed=None, complexity=5, bounds=(0, 0, 15, 15), 
                         safe_zone=(2.0, 2.0, 2.0), overlap_boundaries=True,
                         min_area=1.0, max_area=6.0):
    """Generates a bounded 2D space with randomly placed, irregular polygons."""
    rng = np.random.default_rng(seed)
    walls = []
    
    min_x, min_y, max_x, max_y = bounds
    
    walls.extend([
        [[min_x, min_y], [max_x, min_y]],
        [[max_x, min_y], [max_x, max_y]],
        [[max_x, max_y], [min_x, max_y]],
        [[min_x, max_y], [min_x, min_y]]
    ])
    
    for _ in range(complexity):
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
        max_radius = np.max(final_radii)
        
        valid_position = False
        while not valid_position:
            if overlap_boundaries:
                cx = rng.uniform(min_x, max_x)
                cy = rng.uniform(min_y, max_y)
            else:
                cx = rng.uniform(min_x + max_radius, max_x - max_radius)
                cy = rng.uniform(min_y + max_radius, max_y - max_radius)
            
            dist = np.sqrt((cx - safe_zone[0])**2 + (cy - safe_zone[1])**2)
            if dist > (safe_zone[2] + max_radius):
                valid_position = True
        
        polygon_corners = []
        for angle, r in zip(angles, final_radii):
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            polygon_corners.append([px, py])
            
        for i in range(num_sides):
            p1 = polygon_corners[i]
            p2 = polygon_corners[(i + 1) % num_sides]
            walls.append([p1, p2])
            
    return walls

def apply_sensor_noise(distance, sensor_config, rng):
    noise_config = sensor_config.get("noise", {})
    if not noise_config.get("enabled", False):
        return distance

    distance_ratio = np.clip(distance / sensor_config["max_range"], 0.0, 1.0)
    relative_std = np.interp(
        distance_ratio, [0.0, 1.0],
        [noise_config["min_relative_std"], noise_config["max_relative_std"]]
    )
    noisy_distance = rng.normal(distance, distance * relative_std)
    return np.clip(noisy_distance, 0.0, sensor_config["max_range"])

def choose_avoidance_turn_sign(scan_samples, motion_config, rng):
    front_sector_rad = np.radians(motion_config.get("front_sector_deg", 10.0))
    side_balance_tolerance = motion_config.get("side_balance_tolerance", 0.1)

    left_samples = [(angle, distance) for angle, distance in scan_samples if angle > front_sector_rad]
    right_samples = [(angle, distance) for angle, distance in scan_samples if angle < -front_sector_rad]

    def weighted_clearance(samples):
        if not samples:
            return 0.0

        angles = np.array([angle for angle, _ in samples], dtype=float)
        distances = np.array([distance for _, distance in samples], dtype=float)
        weights = np.cos(angles) ** 2
        return float(np.average(distances, weights=weights))

    left_score = weighted_clearance(left_samples)
    right_score = weighted_clearance(right_samples)

    if abs(left_score - right_score) <= side_balance_tolerance:
        return float(rng.choice((-1.0, 1.0)))
    return 1.0 if left_score > right_score else -1.0

def simulate_lidar(x, y, theta, walls, sensor_config, rng, forward_sector_rad=np.radians(10.0)):
    angles = np.linspace(
        -np.radians(sensor_config["fov_degrees"] / 2),
        np.radians(sensor_config["fov_degrees"] / 2),
        sensor_config["num_rays"],
    )
    measurements = []
    scan_samples = []
    min_dist_forward = float("inf")

    for angle in angles:
        ray_theta = theta + angle
        ray_dir = [np.cos(ray_theta), np.sin(ray_theta)]

        min_dist = sensor_config["max_range"]
        for wall in walls:
            hit_point, dist = get_intersection((x, y), ray_dir, wall)
            if hit_point is not None and dist < min_dist:
                min_dist = dist

        if min_dist < sensor_config["max_range"]:
            noisy_dist = apply_sensor_noise(min_dist, sensor_config, rng)
            measurements.append((angle, noisy_dist))
            scan_samples.append((angle, noisy_dist))
        else:
            scan_samples.append((angle, sensor_config["max_range"]))

        if abs(angle) < forward_sector_rad:
            min_dist_forward = min(min_dist_forward, min_dist)

    return measurements, scan_samples, min_dist_forward

def fit_line_direction(points):
    points_array = np.asarray(points, dtype=float)
    centroid = np.mean(points_array, axis=0)
    centered = points_array - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    return centroid, direction / np.linalg.norm(direction)

def point_line_distance(point, line_point, line_direction):
    offset = point - line_point
    return abs(cross2d(offset, line_direction))

def suppress_nearby_corners(candidates, nms_radius):
    selected = []
    blocked_indices = set()

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        index = candidate["index"]
        if index in blocked_indices:
            continue
        selected.append(candidate)
        blocked_indices.update(range(index - nms_radius, index + nms_radius + 1))

    return sorted(selected, key=lambda item: item["index"])

def extract_landmarks(measurements, config):
    window_size = config["window_size"]
    if len(measurements) < (2 * window_size + 1):
        return []

    local_points = [
        np.array([distance * np.cos(angle), distance * np.sin(angle)], dtype=float)
        for angle, distance in measurements
    ]

    candidates = []
    for index in range(window_size, len(local_points) - window_size):
        left_points = local_points[index - window_size:index]
        right_points = local_points[index + 1:index + 1 + window_size]
        candidate_point = local_points[index]

        left_chain = left_points + [candidate_point]
        right_chain = [candidate_point] + right_points
        if any(np.linalg.norm(b - a) > config["max_neighbor_gap"] for a, b in zip(left_chain, left_chain[1:])):
            continue
        if any(np.linalg.norm(b - a) > config["max_neighbor_gap"] for a, b in zip(right_chain, right_chain[1:])):
            continue

        left_span = np.linalg.norm(left_points[-1] - left_points[0])
        right_span = np.linalg.norm(right_points[-1] - right_points[0])
        if left_span < config["min_segment_span"] or right_span < config["min_segment_span"]:
            continue

        left_center, left_direction = fit_line_direction(left_points)
        right_center, right_direction = fit_line_direction(right_points)

        line_alignment = np.clip(abs(np.dot(left_direction, right_direction)), 0.0, 1.0)
        corner_angle_deg = np.degrees(np.arccos(line_alignment))
        if not (config["min_corner_angle_deg"] <= corner_angle_deg <= config["max_corner_angle_deg"]):
            continue

        left_residual = point_line_distance(candidate_point, left_center, left_direction)
        right_residual = point_line_distance(candidate_point, right_center, right_direction)
        if left_residual > config["max_line_residual"] or right_residual > config["max_line_residual"]:
            continue

        candidates.append({"index": index, "score": corner_angle_deg})

    filtered_candidates = suppress_nearby_corners(candidates, config["nms_radius"])
    return [measurements[candidate["index"]] for candidate in filtered_candidates]

def transform_measurements(measurements, pose):
    points_x, points_y = [], []
    for relative_angle, distance in measurements:
        world_theta = pose[2] + relative_angle
        points_x.append(pose[0] + distance * np.cos(world_theta))
        points_y.append(pose[1] + distance * np.sin(world_theta))
    return points_x, points_y

def measurements_to_world_points(measurements, pose):
    world_points = []
    for relative_angle, distance in measurements:
        world_theta = pose[2] + relative_angle
        world_points.append(
            np.array([pose[0] + distance * np.cos(world_theta), pose[1] + distance * np.sin(world_theta)], dtype=float)
        )
    return world_points

def update_persistent_landmarks(world_points, persistent_landmarks, feature_config):
    persistence_frames = feature_config["persistence_frames"]
    association_radius = feature_config["association_radius"]

    updated_landmarks = []
    for landmark in persistent_landmarks:
        remaining = landmark["ttl"] - 1
        if remaining > 0:
            updated_landmarks.append({"position": landmark["position"], "ttl": remaining})

    for point in world_points:
        best_match = None
        best_distance = association_radius
        for landmark in updated_landmarks:
            distance = np.linalg.norm(point - landmark["position"])
            if distance <= best_distance:
                best_distance = distance
                best_match = landmark

        if best_match is None:
            updated_landmarks.append({"position": point, "ttl": persistence_frames})
        else:
            best_match["position"] = 0.5 * (best_match["position"] + point)
            best_match["ttl"] = persistence_frames

    return updated_landmarks

def get_voxel_key(point_x, point_y, voxel_size):
    return (int(point_x // voxel_size), int(point_y // voxel_size))

def update_voxel_grid(points_x, points_y, voxel_state, voxel_config):
    voxel_size = voxel_config["voxel_size"]
    min_points = voxel_config["min_points_per_voxel"]

    for point_x, point_y in zip(points_x, points_y):
        voxel_key = get_voxel_key(point_x, point_y, voxel_size)
        state = voxel_state[voxel_key]
        state["sum_x"] += point_x
        state["sum_y"] += point_y
        state["count"] += 1

    averaged_points_x, averaged_points_y = [], []
    for state in voxel_state.values():
        if state["count"] >= min_points:
            averaged_points_x.append(state["sum_x"] / state["count"])
            averaged_points_y.append(state["sum_y"] / state["count"])

    return averaged_points_x, averaged_points_y

def ekf_predict(mu, Sigma, measured_distance, measured_turn, noise_config):
    """Step 1 of the EKF: Predict new state and expand covariance."""
    x, y, theta = mu[0], mu[1], mu[2]

    theta_new = theta + measured_turn
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi 

    x_new = x + measured_distance * np.cos(theta_new)
    y_new = y + measured_distance * np.sin(theta_new)

    mu[0], mu[1], mu[2] = x_new, y_new, theta_new

    state_size = len(mu)
    G = np.eye(state_size) 
    G[0, 2] = -measured_distance * np.sin(theta_new)
    G[1, 2] =  measured_distance * np.cos(theta_new)

    var_dist = noise_config["distance_std"] ** 2
    var_turn = np.radians(noise_config["angle_std_deg"]) ** 2
    Q_control = np.array([[var_dist, 0], [0, var_turn]])

    V = np.array([
        [np.cos(theta_new), -measured_distance * np.sin(theta_new)],
        [np.sin(theta_new),  measured_distance * np.cos(theta_new)],
        [0, 1]
    ])

    R_pose = V @ Q_control @ V.T
    R = np.zeros((state_size, state_size))
    R[0:3, 0:3] = R_pose

    Sigma = G @ Sigma @ G.T + R
    return mu, Sigma


# --- Script Execution ---
args = parse_args()
config = load_config(args.config)

simulation_config = config["simulation"]
sensor_config = config["sensor"]
feature_config = config["feature_extraction"]
motion_config = config["motion"]
noise_config = config["odometry_noise"]
voxel_config = config["voxel_grid"]
plot_config = config["plot"]

initial_pose = config["agent"]["initial_pose"]
startup_config = config["agent"]["startup_behavior"]
rng = np.random.default_rng(simulation_config["random_seed"])

# Initialize the global True Pose
true_pose = np.array([
    initial_pose["x"],
    initial_pose["y"],
    np.radians(initial_pose["theta_deg"])
], dtype=float)

# --- Environment Generation ---
env_config = config["environment"]
if env_config.get("generator", {}).get("enabled", False):
    gen_config = env_config["generator"]
    walls = generate_environment(
        seed=simulation_config["random_seed"], 
        complexity=gen_config["complexity"], 
        bounds=(0, 0, gen_config["width"], gen_config["height"]), 
        safe_zone=(initial_pose["x"], initial_pose["y"], 2.0),
        overlap_boundaries=gen_config.get("overlap_boundaries", True),
        min_area=gen_config.get("min_area", 1.0),
        max_area=gen_config.get("max_area", 6.0)
    )
else:
    walls = [tuple(tuple(point) for point in wall) for wall in env_config["walls"]]

startup_rotation_remaining = 0.0
if startup_config.get("enabled", False):
    startup_rotation_remaining = np.radians(max(0.0, startup_config["rotation_degrees"]))

avoiding_obstacle = False
avoidance_turn_sign = 1.0
avoidance_clear_count = 0

# --- EKF Initialization ---
mu = np.array([
    initial_pose["x"],
    initial_pose["y"],
    np.radians(initial_pose["theta_deg"])
], dtype=float)

Sigma = np.zeros((3, 3), dtype=float)

# --- Data Tracking ---
point_cloud_x, point_cloud_y = [], []
voxel_points_x, voxel_points_y = [], []
voxel_state = defaultdict(lambda: {"sum_x": 0.0, "sum_y": 0.0, "count": 0})
persistent_landmarks = []

true_trajectory_x = [true_pose[0]]
true_trajectory_y = [true_pose[1]]
ekf_trajectory_x = [mu[0]]
ekf_trajectory_y = [mu[1]]

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=tuple(plot_config["figsize"]))
ax.set_xlim(*plot_config["xlim"])
ax.set_ylim(*plot_config["ylim"])
ax.set_aspect("equal")
ax.set_title("2D SLAM: Robust Movement & Procedural Caves")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")

for wall in walls:
    ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], "k-", alpha=0.3, linewidth=2)

scatter_cloud = ax.scatter([], [], s=plot_config["point_size"], c="blue", alpha=plot_config["point_alpha"], label="Raw Point Cloud")
voxel_scatter = ax.scatter([], [], s=voxel_config["point_size"], c=voxel_config["color"], alpha=voxel_config["point_alpha"], label="Voxel Grid")
landmark_scatter = ax.scatter([], [], s=80, facecolors='none', edgecolors='red', linewidths=2, label="Detected Landmarks")

true_traj_line, = ax.plot([], [], "g--", alpha=0.8, label="True Trajectory")
ekf_traj_line, = ax.plot([], [], "m-", alpha=0.7, label="EKF Trajectory")
true_marker, = ax.plot([], [], "ro", markersize=6, label="True Sensor")
ekf_marker, = ax.plot([], [], "mx", markersize=8, label="EKF Pose")
true_heading, = ax.plot([], [], "r-", linewidth=2)
ekf_heading, = ax.plot([], [], "m-", linewidth=2)

ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

def update(frame):
    global mu, Sigma, startup_rotation_remaining, persistent_landmarks, true_pose
    global avoiding_obstacle, avoidance_turn_sign, avoidance_clear_count

    observation_pose = true_pose.copy()
    forward_sector_rad = np.radians(motion_config.get("front_sector_deg", 10.0))
    measurements, scan_samples, dist_ahead = simulate_lidar(
        observation_pose[0], observation_pose[1], observation_pose[2],
        walls, sensor_config, rng, forward_sector_rad,
    )

    observed_landmarks = extract_landmarks(measurements, feature_config)
    observed_landmark_points = measurements_to_world_points(observed_landmarks, observation_pose)
    persistent_landmarks = update_persistent_landmarks(
        observed_landmark_points, persistent_landmarks, feature_config,
    )

    mapped_x, mapped_y = transform_measurements(measurements, mu)
    point_cloud_x.extend(mapped_x)
    point_cloud_y.extend(mapped_y)

    averaged_x, averaged_y = update_voxel_grid(mapped_x, mapped_y, voxel_state, voxel_config)
    voxel_points_x[:] = averaged_x
    voxel_points_y[:] = averaged_y

    # --- Robust Movement Logic ---
    collision_turn = np.radians(motion_config["collision_turn_deg"])
    collision_distance = motion_config["collision_distance"]
    clear_distance = motion_config.get("clear_distance", collision_distance * 1.4)
    clearance_frames = max(1, int(motion_config.get("clearance_frames", 3)))
    obstacle_ahead = dist_ahead < collision_distance

    if startup_rotation_remaining > 0:
        commanded_turn = min(np.radians(startup_config["turn_step_deg"]), startup_rotation_remaining)
        commanded_distance = 0.0
        startup_rotation_remaining -= commanded_turn
    else:
        if obstacle_ahead and not avoiding_obstacle:
            avoidance_turn_sign = choose_avoidance_turn_sign(scan_samples, motion_config, rng)
            avoiding_obstacle = True
            avoidance_clear_count = 0

        if avoiding_obstacle:
            if dist_ahead > clear_distance:
                avoidance_clear_count += 1
                if avoidance_clear_count >= clearance_frames:
                    avoiding_obstacle = False
                    avoidance_clear_count = 0
            else:
                avoidance_clear_count = 0

        if avoiding_obstacle:
            commanded_turn = avoidance_turn_sign * collision_turn
            commanded_distance = 0.0
        else:
            commanded_turn = np.radians(rng.uniform(-motion_config["random_turn_deg"], motion_config["random_turn_deg"]))
            commanded_distance = motion_config["linear_speed"]

    proposed_theta = true_pose[2] + commanded_turn
    proposed_x = true_pose[0] + commanded_distance * np.cos(proposed_theta)
    proposed_y = true_pose[1] + commanded_distance * np.sin(proposed_theta)

    # Physics Bumper Check
    if not is_path_clear(true_pose[0], true_pose[1], proposed_x, proposed_y, walls):
        commanded_distance = 0.0
        if not avoiding_obstacle:
            avoidance_turn_sign = choose_avoidance_turn_sign(scan_samples, motion_config, rng)
            avoiding_obstacle = True
        avoidance_clear_count = 0
        commanded_turn = avoidance_turn_sign * collision_turn
        proposed_theta = true_pose[2] + commanded_turn

    # Commit true movement
    true_pose[2] = proposed_theta
    true_pose[0] += commanded_distance * np.cos(true_pose[2])
    true_pose[1] += commanded_distance * np.sin(true_pose[2])

    # Noisy Odometry
    measured_turn = commanded_turn + np.radians(rng.normal(0.0, noise_config["angle_std_deg"]))
    if commanded_distance == 0.0:
        measured_distance = 0.0
    else:
        measured_distance = max(0.0, commanded_distance + rng.normal(0.0, noise_config["distance_std"]))

    # EKF Predict
    mu, Sigma = ekf_predict(mu, Sigma, measured_distance, measured_turn, noise_config)

    true_trajectory_x.append(true_pose[0])
    true_trajectory_y.append(true_pose[1])
    ekf_trajectory_x.append(mu[0])
    ekf_trajectory_y.append(mu[1])

    # Update Visuals
    scatter_cloud.set_offsets(np.c_[point_cloud_x, point_cloud_y])
    voxel_scatter.set_offsets(np.c_[voxel_points_x, voxel_points_y])
    
    if persistent_landmarks:
        landmark_positions = np.array([landmark["position"] for landmark in persistent_landmarks], dtype=float)
        landmark_alphas = np.array([landmark["ttl"] / feature_config["persistence_frames"] for landmark in persistent_landmarks], dtype=float)
        landmark_edgecolors = np.column_stack([
            np.ones(len(persistent_landmarks)), np.zeros(len(persistent_landmarks)), 
            np.zeros(len(persistent_landmarks)), landmark_alphas
        ])
        landmark_scatter.set_offsets(landmark_positions)
        landmark_scatter.set_edgecolors(landmark_edgecolors)
        landmark_scatter.set_facecolors(np.zeros((len(persistent_landmarks), 4)))
    else:
        landmark_scatter.set_offsets(np.empty((0, 2)))
        landmark_scatter.set_edgecolors(np.empty((0, 4)))
        landmark_scatter.set_facecolors(np.empty((0, 4)))

    true_traj_line.set_data(true_trajectory_x, true_trajectory_y)
    ekf_traj_line.set_data(ekf_trajectory_x, ekf_trajectory_y)
    true_marker.set_data([true_pose[0]], [true_pose[1]])
    ekf_marker.set_data([mu[0]], [mu[1]])

    heading_length = 0.5
    true_heading.set_data(
        [true_pose[0], true_pose[0] + heading_length * np.cos(true_pose[2])],
        [true_pose[1], true_pose[1] + heading_length * np.sin(true_pose[2])],
    )
    ekf_heading.set_data(
        [mu[0], mu[0] + heading_length * np.cos(mu[2])],
        [mu[1], mu[1] + heading_length * np.sin(mu[2])],
    )

    return (
        scatter_cloud, voxel_scatter, landmark_scatter,
        true_traj_line, ekf_traj_line, true_marker, ekf_marker,
        true_heading, ekf_heading,
    )

ani = animation.FuncAnimation(
    fig, update, frames=simulation_config["frames"],
    interval=max(0, simulation_config["delay_ms"]), blit=False,
)
plt.show()
