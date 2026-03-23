from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .slam import track_display_value
from .sim import AppConfig, SimulationState, create_simulation, step_simulation


@dataclass
class PlotArtists:
    fig: object
    ax: object
    occupancy_image: object
    scatter_cloud: object
    voxel_scatter: object
    landmark_scatter: object
    track_scatter: object
    true_traj_line: object
    ekf_traj_line: object
    true_marker: object
    ekf_marker: object
    true_heading: object
    ekf_heading: object
    ekf_text: object
    status_text: object


def configure_matplotlib_backend():
    import matplotlib

    if sys.platform.startswith("linux"):
        if "QT_QPA_PLATFORM" not in os.environ and os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"

        has_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        matplotlib.use("qtagg" if has_display else "Agg")
        return

    matplotlib.use("qtagg")


def import_matplotlib_modules():
    configure_matplotlib_backend()
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    return animation, plt


def _get_environment_size_text(state: SimulationState):
    if state.config.environment.generator.enabled:
        width = state.config.environment.generator.width
        height = state.config.environment.generator.height
    else:
        if not state.walls:
            width = 0.0
            height = 0.0
        else:
            xs = [coordinate for wall in state.walls for coordinate in (wall.start[0], wall.end[0])]
            ys = [coordinate for wall in state.walls for coordinate in (wall.start[1], wall.end[1])]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

    return f"env={width:.1f}x{height:.1f}"


def _build_status_text(state: SimulationState):
    sensor = state.config.sensor
    return (
        f"fov={sensor.fov_degrees:g} deg"
        f" | rays={sensor.num_rays}"
        f" | max_range={sensor.max_range:g}"
        f" | {_get_environment_size_text(state)}"
        " | space: pause/resume"
    )


def _build_ekf_diagnostics_text(state: SimulationState):
    if state.last_step_result is None:
        return "EKF: waiting for first step"

    diagnostics = state.last_step_result.ekf_diagnostics
    debug_info = state.last_step_result.ekf_debug_info
    association_result = state.last_step_result.association_result
    track_count = state.last_step_result.landmark_track_count
    augmented_count = len(state.last_step_result.augmented_landmark_track_ids)
    return (
        f"EKF={state.config.ekf.mode}"
        f" | state={len(state.slam_state.mu)}"
        f" | Tracks={track_count}"
        f" | aug={augmented_count}"
        f" | Assoc={association_result.method}"
        f" m={len(association_result.matched)}"
        f" u={len(association_result.unmatched_observations)}"
        f" r={len(association_result.rejected)}"
        f" a={len(association_result.ambiguous)}"
        "\n"
        f"EKF diag | cand={diagnostics.num_candidate_observations}"
        f" match={diagnostics.num_matches}"
        f" rej={diagnostics.num_rejections}"
        f" amb={diagnostics.ambiguous_rejections}"
        f" | trSigma={diagnostics.trace_sigma_before:.4f}->{diagnostics.trace_sigma_after:.4f}"
        f" | dpose={diagnostics.pose_update_norm:.4f}"
        f" | NIS(mean/max)={diagnostics.mean_nis:.3f}/{diagnostics.max_nis:.3f}"
        f"\nEKF std | x={debug_info.pose_std_x:.3f}"
        f" y={debug_info.pose_std_y:.3f}"
        f" theta={np.degrees(debug_info.pose_std_theta):.2f} deg"
    )


def _track_colors(state: SimulationState):
    import matplotlib.pyplot as plt

    tracks = list(state.slam_state.landmark_track_state.tracks.values())
    if not tracks:
        return np.empty((0, 4))

    mode = state.config.plot.track_color_mode
    if mode == "augmented":
        return np.array(
            [
                (0.0, 0.6, 0.2, 0.95) if track_display_value(track, mode, state.step_index, state.slam_state.ekf_slam_index) > 0.5
                else (0.45, 0.45, 0.45, 0.55)
                for track in tracks
            ],
            dtype=float,
        )

    values = np.array(
        [
            track_display_value(track, mode, state.step_index, state.slam_state.ekf_slam_index)
            for track in tracks
        ],
        dtype=float,
    )
    if mode == "quality":
        normalized = np.clip(values, 0.0, 1.0)
        cmap = plt.get_cmap("viridis")
    else:
        max_value = max(float(np.max(values)), 1.0)
        normalized = np.clip(values / max_value, 0.0, 1.0)
        cmap = plt.get_cmap("plasma_r")
    return np.asarray(cmap(normalized), dtype=float)


def _build_occupancy_rgba(state: SimulationState):
    occupancy_config = state.config.occupancy_grid
    grid_state = state.slam_state.occupancy_grid_state
    log_odds = np.asarray(grid_state.log_odds, dtype=float)

    if log_odds.size == 0:
        return np.zeros((0, 0, 4), dtype=float)

    probabilities = 1.0 / (1.0 + np.exp(-log_odds))
    rgba = np.zeros(log_odds.shape + (4,), dtype=float)
    evidence_scale = max(abs(occupancy_config.log_odds_min), abs(occupancy_config.log_odds_max), 1e-9)
    evidence = np.clip(np.abs(log_odds) / evidence_scale, 0.0, 1.0)

    free_mask = probabilities <= occupancy_config.free_threshold
    occupied_mask = probabilities >= occupancy_config.occupied_threshold
    unknown_mask = ~(free_mask | occupied_mask)

    rgba[free_mask, 0] = 1.0
    rgba[free_mask, 1] = 1.0
    rgba[free_mask, 2] = 1.0
    rgba[free_mask, 3] = 0.08 + 0.20 * evidence[free_mask]

    rgba[occupied_mask, 0] = 0.12
    rgba[occupied_mask, 1] = 0.12
    rgba[occupied_mask, 2] = 0.12
    rgba[occupied_mask, 3] = 0.15 + 0.70 * evidence[occupied_mask]

    rgba[unknown_mask, 0] = 0.65
    rgba[unknown_mask, 1] = 0.65
    rgba[unknown_mask, 2] = 0.65
    rgba[unknown_mask, 3] = 0.06 * evidence[unknown_mask]

    return rgba


def create_plot(state: SimulationState):
    _, plt = import_matplotlib_modules()
    fig, ax = plt.subplots(figsize=state.config.plot.figsize)
    ax.set_xlim(*state.config.plot.xlim)
    ax.set_ylim(*state.config.plot.ylim)
    ax.set_aspect("equal")
    ax.set_title("2D SLAM: Robust Movement & Procedural Caves")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    occupancy_state = state.slam_state.occupancy_grid_state
    occupancy_extent = (
        occupancy_state.origin_x,
        occupancy_state.origin_x + occupancy_state.width_cells * occupancy_state.cell_size,
        occupancy_state.origin_y,
        occupancy_state.origin_y + occupancy_state.height_cells * occupancy_state.cell_size,
    )
    occupancy_image = ax.imshow(
        _build_occupancy_rgba(state),
        origin="lower",
        extent=occupancy_extent,
        interpolation="nearest",
        zorder=0,
    )
    occupancy_image.set_visible(state.config.plot.show_occupancy_grid and state.config.occupancy_grid.enabled)

    for wall in state.walls:
        ax.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]], "k-", alpha=0.3, linewidth=2, zorder=3)

    scatter_cloud = ax.scatter([], [], s=state.config.plot.point_size, c="blue", alpha=state.config.plot.point_alpha, label="Raw Point Cloud", zorder=1)
    voxel_scatter = ax.scatter([], [], s=state.config.voxel_grid.point_size, c=state.config.voxel_grid.color, alpha=state.config.voxel_grid.point_alpha, label="Voxel Grid", zorder=2)
    landmark_scatter = ax.scatter([], [], s=80, facecolors="none", edgecolors="red", linewidths=2, label="Detected Landmarks", zorder=4)
    track_scatter = ax.scatter([], [], s=40, linewidths=0.5, edgecolors="black", alpha=0.95, label="Landmark Tracks", zorder=4)
    scatter_cloud.set_visible(state.config.plot.show_point_cloud)
    voxel_scatter.set_visible(state.config.plot.show_voxel_grid)
    track_scatter.set_visible(state.config.plot.show_landmark_tracks)

    true_traj_line, = ax.plot([], [], "g--", alpha=0.8, label="True Trajectory")
    ekf_traj_line, = ax.plot([], [], "m-", alpha=0.7, label="EKF Trajectory")
    true_marker, = ax.plot([], [], "ro", markersize=6, label="True Sensor")
    ekf_marker, = ax.plot([], [], "mx", markersize=8, label="EKF Pose")
    true_heading, = ax.plot([], [], "r-", linewidth=2)
    ekf_heading, = ax.plot([], [], "m-", linewidth=2)
    ekf_text = ax.text(
        0.02,
        0.98,
        _build_ekf_diagnostics_text(state),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.6"},
    )
    ekf_text.set_visible(state.config.plot.show_ekf_overlay)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    status_text = fig.text(0.5, 0.04, _build_status_text(state), ha="center", va="center")
    plt.tight_layout(rect=(0, 0.12, 1, 1))

    return PlotArtists(
        fig=fig,
        ax=ax,
        occupancy_image=occupancy_image,
        scatter_cloud=scatter_cloud,
        voxel_scatter=voxel_scatter,
        landmark_scatter=landmark_scatter,
        track_scatter=track_scatter,
        true_traj_line=true_traj_line,
        ekf_traj_line=ekf_traj_line,
        true_marker=true_marker,
        ekf_marker=ekf_marker,
        true_heading=true_heading,
        ekf_heading=ekf_heading,
        ekf_text=ekf_text,
        status_text=status_text,
    )


def render_simulation(state: SimulationState):
    if state.artists is None:
        raise RuntimeError("Simulation artists have not been created. Call create_plot() first.")

    artists = state.artists
    slam_state = state.slam_state
    agent_state = state.agent_state

    artists.scatter_cloud.set_offsets(np.c_[slam_state.point_cloud_x, slam_state.point_cloud_y])
    artists.scatter_cloud.set_visible(state.config.plot.show_point_cloud)
    artists.voxel_scatter.set_offsets(np.c_[slam_state.voxel_points_x, slam_state.voxel_points_y])
    artists.voxel_scatter.set_visible(state.config.plot.show_voxel_grid)
    artists.occupancy_image.set_data(_build_occupancy_rgba(state))
    artists.occupancy_image.set_visible(state.config.plot.show_occupancy_grid and state.config.occupancy_grid.enabled)

    if slam_state.persistent_landmarks:
        landmark_positions = np.array([landmark.position for landmark in slam_state.persistent_landmarks], dtype=float)
        landmark_alphas = np.array(
            [landmark.ttl / state.config.feature_extraction.persistence_frames for landmark in slam_state.persistent_landmarks],
            dtype=float,
        )
        landmark_edgecolors = np.column_stack(
            [
                np.ones(len(slam_state.persistent_landmarks)),
                np.zeros(len(slam_state.persistent_landmarks)),
                np.zeros(len(slam_state.persistent_landmarks)),
                landmark_alphas,
            ]
        )
        artists.landmark_scatter.set_offsets(landmark_positions)
        artists.landmark_scatter.set_edgecolors(landmark_edgecolors)
        artists.landmark_scatter.set_facecolors(np.zeros((len(slam_state.persistent_landmarks), 4)))
    else:
        artists.landmark_scatter.set_offsets(np.empty((0, 2)))
        artists.landmark_scatter.set_edgecolors(np.empty((0, 4)))
        artists.landmark_scatter.set_facecolors(np.empty((0, 4)))

    if state.config.plot.show_landmark_tracks:
        track_positions = np.array(
            [track.position for track in slam_state.landmark_track_state.tracks.values()],
            dtype=float,
        )
        artists.track_scatter.set_offsets(track_positions if len(track_positions) else np.empty((0, 2)))
        artists.track_scatter.set_facecolors(_track_colors(state))
        artists.track_scatter.set_visible(True)
    else:
        artists.track_scatter.set_offsets(np.empty((0, 2)))
        artists.track_scatter.set_facecolors(np.empty((0, 4)))
        artists.track_scatter.set_visible(False)

    artists.true_traj_line.set_data(slam_state.true_trajectory_x, slam_state.true_trajectory_y)
    artists.ekf_traj_line.set_data(slam_state.ekf_trajectory_x, slam_state.ekf_trajectory_y)
    artists.true_marker.set_data([agent_state.true_pose[0]], [agent_state.true_pose[1]])
    artists.ekf_marker.set_data([slam_state.mu[0]], [slam_state.mu[1]])

    heading_length = 0.5
    artists.true_heading.set_data(
        [agent_state.true_pose[0], agent_state.true_pose[0] + heading_length * np.cos(agent_state.true_pose[2])],
        [agent_state.true_pose[1], agent_state.true_pose[1] + heading_length * np.sin(agent_state.true_pose[2])],
    )
    artists.ekf_heading.set_data(
        [slam_state.mu[0], slam_state.mu[0] + heading_length * np.cos(slam_state.mu[2])],
        [slam_state.mu[1], slam_state.mu[1] + heading_length * np.sin(slam_state.mu[2])],
    )
    if state.config.plot.show_ekf_overlay:
        artists.ekf_text.set_text(_build_ekf_diagnostics_text(state))
    pause_suffix = " [Paused]" if state.is_paused else ""
    artists.status_text.set_text(_build_status_text(state) + pause_suffix)

    return (
        artists.occupancy_image,
        artists.scatter_cloud,
        artists.voxel_scatter,
        artists.landmark_scatter,
        artists.track_scatter,
        artists.true_traj_line,
        artists.ekf_traj_line,
        artists.true_marker,
        artists.ekf_marker,
        artists.true_heading,
        artists.ekf_heading,
        artists.ekf_text,
        artists.status_text,
    )


def update_frame(frame, state: SimulationState):
    del frame
    step_simulation(state)
    return render_simulation(state)


def run_simulation(config: AppConfig | Mapping[str, object]):
    animation, plt = import_matplotlib_modules()
    state = create_simulation(config)
    state.artists = create_plot(state)

    def update(frame):
        return update_frame(frame, state)

    state.animation = animation.FuncAnimation(
        state.artists.fig,
        update,
        frames=state.config.simulation.frames,
        interval=max(0, state.config.simulation.delay_ms),
        blit=False,
    )

    def on_key_press(event):
        if event.key not in {" ", "space"} or state.animation is None:
            return

        state.is_paused = not state.is_paused
        if state.is_paused:
            state.animation.event_source.stop()
        else:
            state.animation.event_source.start()
        render_simulation(state)
        state.artists.fig.canvas.draw_idle()

    state.artists.fig.canvas.mpl_connect("key_press_event", on_key_press)
    plt.show()
    return state
