import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cave_slam import DEFAULT_CONFIG_PATH, create_simulation, load_config, step_simulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a headless cave SLAM experiment for a fixed number of steps."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of headless simulation steps to run.",
    )
    parser.add_argument(
        "--summary-every",
        type=int,
        default=0,
        help="Print an intermediate summary every N steps. Use 0 to disable.",
    )
    return parser.parse_args()


def format_pose(pose):
    return f"x={pose[0]:.3f}, y={pose[1]:.3f}, theta_deg={pose[2] * 180.0 / 3.141592653589793:.2f}"


def main():
    args = parse_args()
    config = load_config(args.config)
    state = create_simulation(config)

    last_result = None
    for step_index in range(1, args.steps + 1):
        last_result = step_simulation(state)
        if args.summary_every > 0 and step_index % args.summary_every == 0:
            print(
                f"step={step_index} "
                f"true_pose=({format_pose(state.agent_state.true_pose)}) "
                f"ekf_pose=({format_pose(state.slam_state.mu)}) "
                f"landmarks={len(state.slam_state.persistent_landmarks)} "
                f"raw_points={len(state.slam_state.point_cloud_x)}"
            )

    print("Headless experiment complete")
    print(f"steps={args.steps}")
    print(f"true_pose=({format_pose(state.agent_state.true_pose)})")
    print(f"ekf_pose=({format_pose(state.slam_state.mu)})")
    print(f"persistent_landmarks={len(state.slam_state.persistent_landmarks)}")
    print(f"raw_point_cloud_points={len(state.slam_state.point_cloud_x)}")
    print(f"voxel_points={len(state.slam_state.voxel_points_x)}")
    if last_result is not None:
        print(f"last_scan_hits={len(last_result.lidar_scan.measurements)}")
        print(f"last_command_distance={last_result.motion_command.distance:.3f}")
        print(f"last_command_turn_deg={last_result.motion_command.turn * 180.0 / 3.141592653589793:.3f}")


if __name__ == "__main__":
    main()
