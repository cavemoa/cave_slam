from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .sim import FeatureExtractionConfig, MeasurementModelConfig, OdometryNoiseConfig, SensorConfig, VoxelGridConfig


@dataclass(frozen=True)
class WallSegment:
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class ScanMeasurement:
    angle: float
    distance: float


@dataclass
class PersistentLandmark:
    position: np.ndarray
    ttl: int


@dataclass
class LandmarkTrack:
    track_id: int
    position: np.ndarray
    covariance: np.ndarray | None
    observation_count: int
    last_seen_frame: int
    ttl: int
    quality_score: float


@dataclass
class LandmarkTrackState:
    tracks: dict[int, LandmarkTrack]
    next_track_id: int = 0


@dataclass(frozen=True)
class TrackAssignment:
    observation_index: int
    track_id: int
    created: bool


@dataclass
class TrackUpdateResult:
    track_state: LandmarkTrackState
    assignments: tuple[TrackAssignment, ...]


@dataclass
class EkfSlamStateIndex:
    track_id_to_index: dict[int, int]
    state_track_ids: list[int]


@dataclass
class LidarScan:
    measurements: list[ScanMeasurement]
    scan_samples: list[ScanMeasurement]
    min_dist_forward: float


@dataclass(frozen=True)
class EkfDebugInfo:
    trace_sigma: float
    pose_std_x: float
    pose_std_y: float
    pose_std_theta: float


@dataclass(frozen=True)
class LandmarkObservation:
    range: float
    bearing: float
    world_position: np.ndarray | None = None
    source_id: int | None = None


@dataclass(frozen=True)
class AssociationCandidate:
    observation_index: int
    track_id: int
    innovation: np.ndarray
    distance: float
    mahalanobis_distance: float


@dataclass(frozen=True)
class AssociationMatch:
    observation_index: int
    track_id: int
    innovation: np.ndarray
    distance: float
    mahalanobis_distance: float


@dataclass(frozen=True)
class AssociationResult:
    matched: tuple[AssociationMatch, ...]
    unmatched_observations: tuple[int, ...]
    rejected: tuple[AssociationCandidate, ...]
    ambiguous: tuple[AssociationCandidate, ...]
    method: str
    gating_applied: bool


@dataclass(frozen=True)
class TruthObservationSet:
    observations: list[LandmarkObservation]
    landmark_positions: list[np.ndarray]


@dataclass(frozen=True)
class EkfUpdateResult:
    mu: np.ndarray
    Sigma: np.ndarray
    innovation: np.ndarray
    kalman_gain: np.ndarray
    nis: float


@dataclass(frozen=True)
class InnovationStats:
    innovation_norm: float
    nis: float
    accepted: bool


@dataclass(frozen=True)
class EkfAssociationSummary:
    num_candidate_observations: int
    num_matches: int
    num_rejections: int
    source: str
    gating_applied: bool


@dataclass(frozen=True)
class EkfStepDiagnostics:
    num_candidate_observations: int
    num_matches: int
    num_rejections: int
    ambiguous_rejections: int
    trace_sigma_before: float
    trace_sigma_after: float
    pose_update_norm: float
    innovation_stats: tuple[InnovationStats, ...]
    mean_nis: float
    max_nis: float


@dataclass
class VoxelCellState:
    sum_w: float = 0.0
    sum_wx: float = 0.0
    sum_wy: float = 0.0
    count: float = 0.0
    best_x: float = 0.0
    best_y: float = 0.0
    best_weight: float = 0.0
    best_distance: float = float("inf")
    last_frame: int = 0


def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def normalize_angle(angle: float):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def normalize_state_angle(mu: np.ndarray):
    mu[2] = normalize_angle(mu[2])
    return mu


def symmetrize_covariance(Sigma: np.ndarray):
    return 0.5 * (Sigma + Sigma.T)


def ensure_positive_semidefinite(Sigma: np.ndarray, min_eigenvalue: float = 1e-9):
    Sigma = symmetrize_covariance(Sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    clipped_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    Sigma_psd = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    return symmetrize_covariance(Sigma_psd)


def compute_ekf_debug_info(Sigma: np.ndarray):
    pose_covariance = symmetrize_covariance(Sigma[:3, :3])
    diagonal = np.clip(np.diag(pose_covariance), 0.0, None)
    return EkfDebugInfo(
        trace_sigma=float(np.trace(pose_covariance)),
        pose_std_x=float(np.sqrt(diagonal[0])),
        pose_std_y=float(np.sqrt(diagonal[1])),
        pose_std_theta=float(np.sqrt(diagonal[2])),
    )


def compute_pose_delta_norm(previous_mu: np.ndarray, updated_mu: np.ndarray):
    delta = np.asarray(updated_mu, dtype=float)[:3] - np.asarray(previous_mu, dtype=float)[:3]
    delta[2] = normalize_angle(float(delta[2]))
    return float(np.linalg.norm(delta))


def build_innovation_stats(update_results: Sequence[EkfUpdateResult]):
    return tuple(
        InnovationStats(
            innovation_norm=float(np.linalg.norm(result.innovation)),
            nis=float(result.nis),
            accepted=True,
        )
        for result in update_results
    )


def is_nis_accepted(nis: float, nis_threshold: float | None):
    if nis_threshold is None:
        return True
    return float(nis) <= float(nis_threshold)


def build_ekf_step_diagnostics(
    pose_before_update: np.ndarray,
    pose_after_update: np.ndarray,
    sigma_before_update: np.ndarray,
    sigma_after_update: np.ndarray,
    update_results: Sequence[EkfUpdateResult],
    num_candidate_observations: int,
    num_rejections: int = 0,
    ambiguous_rejections: int = 0,
):
    innovation_stats = build_innovation_stats(update_results)
    nis_values = [stats.nis for stats in innovation_stats]
    num_matches = len(update_results)

    return EkfStepDiagnostics(
        num_candidate_observations=num_candidate_observations,
        num_matches=num_matches,
        num_rejections=num_rejections,
        ambiguous_rejections=ambiguous_rejections,
        trace_sigma_before=float(np.trace(symmetrize_covariance(np.asarray(sigma_before_update, dtype=float)[:3, :3]))),
        trace_sigma_after=float(np.trace(symmetrize_covariance(np.asarray(sigma_after_update, dtype=float)[:3, :3]))),
        pose_update_norm=compute_pose_delta_norm(pose_before_update, pose_after_update),
        innovation_stats=innovation_stats,
        mean_nis=float(np.mean(nis_values)) if nis_values else 0.0,
        max_nis=float(np.max(nis_values)) if nis_values else 0.0,
    )


def predict_range_bearing(mu: np.ndarray, landmark_position: np.ndarray):
    dx = float(landmark_position[0] - mu[0])
    dy = float(landmark_position[1] - mu[1])
    squared_range = dx * dx + dy * dy
    if squared_range < 1e-12:
        raise ValueError("Landmark is too close to the robot pose for a stable range-bearing prediction.")

    predicted_range = float(np.sqrt(squared_range))
    predicted_bearing = normalize_angle(float(np.arctan2(dy, dx) - mu[2]))
    return np.array([predicted_range, predicted_bearing], dtype=float)


def range_bearing_jacobian_pose(mu: np.ndarray, landmark_position: np.ndarray):
    dx = float(landmark_position[0] - mu[0])
    dy = float(landmark_position[1] - mu[1])
    squared_range = dx * dx + dy * dy
    if squared_range < 1e-12:
        raise ValueError("Landmark is too close to the robot pose for a stable measurement Jacobian.")

    predicted_range = float(np.sqrt(squared_range))
    return np.array(
        [
            [-dx / predicted_range, -dy / predicted_range, 0.0],
            [dy / squared_range, -dx / squared_range, -1.0],
        ],
        dtype=float,
    )


def measurement_noise_matrix(config: MeasurementModelConfig):
    if config.model_type != "range_bearing":
        raise ValueError(f"Unsupported measurement model type: {config.model_type}")

    range_variance = config.range_std ** 2
    bearing_variance = np.radians(config.bearing_std_deg) ** 2
    return np.array([[range_variance, 0.0], [0.0, bearing_variance]], dtype=float)


def innovation_range_bearing(z: np.ndarray, z_hat: np.ndarray):
    innovation = np.asarray(z, dtype=float) - np.asarray(z_hat, dtype=float)
    innovation[1] = normalize_angle(float(innovation[1]))
    return innovation


def observation_to_measurement_vector(observation: LandmarkObservation):
    return np.array([observation.range, observation.bearing], dtype=float)


def initialize_ekf_slam_state_index():
    return EkfSlamStateIndex(track_id_to_index={}, state_track_ids=[])


def get_landmark_state_index(state_index: EkfSlamStateIndex, track_id: int):
    return state_index.track_id_to_index.get(track_id)


def get_landmark_state_position(
    mu: np.ndarray,
    track: LandmarkTrack,
    state_index: EkfSlamStateIndex | None = None,
):
    if state_index is None:
        return np.asarray(track.position, dtype=float)

    landmark_index = get_landmark_state_index(state_index, track.track_id)
    if landmark_index is None:
        return np.asarray(track.position, dtype=float)
    return np.asarray(mu[landmark_index:landmark_index + 2], dtype=float)


def compute_mahalanobis_distance(innovation: np.ndarray, innovation_covariance: np.ndarray):
    innovation_covariance = symmetrize_covariance(np.asarray(innovation_covariance, dtype=float))
    inverse_covariance = np.linalg.inv(innovation_covariance)
    innovation_vector = np.asarray(innovation, dtype=float)
    return float(innovation_vector.T @ inverse_covariance @ innovation_vector)


def _candidate_distance(observation: LandmarkObservation, track: LandmarkTrack):
    if observation.world_position is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(observation.world_position, dtype=float) - track.position))


def _build_empty_association_result(method: str, gating_applied: bool):
    return AssociationResult(
        matched=(),
        unmatched_observations=(),
        rejected=(),
        ambiguous=(),
        method=method,
        gating_applied=gating_applied,
    )


def is_association_ambiguous(
    best_score: float,
    second_best_score: float | None,
    ambiguity_ratio_threshold: float,
    ambiguity_margin_threshold: float,
):
    if second_best_score is None or not np.isfinite(second_best_score):
        return False

    margin = float(second_best_score - best_score)
    if margin <= ambiguity_margin_threshold:
        return True

    safe_best_score = max(float(best_score), 1e-9)
    score_ratio = float(second_best_score / safe_best_score)
    return score_ratio <= ambiguity_ratio_threshold


def associate_landmarks_nearest_neighbor(
    observations: Sequence[LandmarkObservation],
    track_state: LandmarkTrackState,
    max_distance: float,
    min_track_quality: float,
    ambiguity_ratio_threshold: float,
    ambiguity_margin_threshold: float,
    mu: np.ndarray | None = None,
    state_index: EkfSlamStateIndex | None = None,
):
    if not observations:
        return _build_empty_association_result(method="nearest_neighbor", gating_applied=False)

    unmatched_observations: list[int] = []
    matched: list[AssociationMatch] = []
    rejected: list[AssociationCandidate] = []
    ambiguous: list[AssociationCandidate] = []
    used_track_ids: set[int] = set()

    for observation_index, observation in enumerate(observations):
        best_track = None
        best_distance = max_distance
        second_best_distance: float | None = None

        for track in track_state.tracks.values():
            if track.track_id in used_track_ids or track.quality_score < min_track_quality:
                continue
            if observation.world_position is None:
                continue

            landmark_position = get_landmark_state_position(
                np.asarray(mu, dtype=float) if mu is not None else track.position,
                track,
                state_index,
            )
            distance = float(np.linalg.norm(np.asarray(observation.world_position, dtype=float) - landmark_position))
            if distance <= best_distance:
                second_best_distance = best_distance if best_track is not None else second_best_distance
                best_distance = distance
                best_track = track
            elif second_best_distance is None or distance < second_best_distance:
                second_best_distance = distance

        if best_track is None:
            unmatched_observations.append(observation_index)
            continue

        candidate = AssociationCandidate(
            observation_index=observation_index,
            track_id=best_track.track_id,
            innovation=np.zeros(2, dtype=float),
            distance=best_distance,
            mahalanobis_distance=0.0,
        )
        if is_association_ambiguous(
            best_score=best_distance,
            second_best_score=second_best_distance,
            ambiguity_ratio_threshold=ambiguity_ratio_threshold,
            ambiguity_margin_threshold=ambiguity_margin_threshold,
        ):
            ambiguous.append(candidate)
            unmatched_observations.append(observation_index)
            continue

        matched.append(
            AssociationMatch(
                observation_index=candidate.observation_index,
                track_id=candidate.track_id,
                innovation=np.array(candidate.innovation, dtype=float, copy=True),
                distance=candidate.distance,
                mahalanobis_distance=candidate.mahalanobis_distance,
            )
        )
        used_track_ids.add(best_track.track_id)

    return AssociationResult(
        matched=tuple(matched),
        unmatched_observations=tuple(unmatched_observations),
        rejected=tuple(rejected),
        ambiguous=tuple(ambiguous),
        method="nearest_neighbor",
        gating_applied=False,
    )


def associate_landmarks_mahalanobis(
    observations: Sequence[LandmarkObservation],
    track_state: LandmarkTrackState,
    mu: np.ndarray,
    Sigma: np.ndarray,
    measurement_config: MeasurementModelConfig,
    max_distance: float,
    mahalanobis_threshold: float,
    min_track_quality: float,
    ambiguity_ratio_threshold: float,
    ambiguity_margin_threshold: float,
    state_index: EkfSlamStateIndex | None = None,
):
    if not observations:
        return _build_empty_association_result(method="mahalanobis", gating_applied=True)

    unmatched_observations: list[int] = []
    matched: list[AssociationMatch] = []
    rejected: list[AssociationCandidate] = []
    ambiguous: list[AssociationCandidate] = []
    used_track_ids: set[int] = set()

    for observation_index, observation in enumerate(observations):
        z = observation_to_measurement_vector(observation)
        best_candidate: AssociationCandidate | None = None
        second_best_candidate: AssociationCandidate | None = None

        for track in track_state.tracks.values():
            if track.track_id in used_track_ids or track.quality_score < min_track_quality:
                continue

            landmark_position = get_landmark_state_position(mu, track, state_index)
            candidate_distance = (
                float(np.linalg.norm(np.asarray(observation.world_position, dtype=float) - landmark_position))
                if observation.world_position is not None
                else float("inf")
            )
            if np.isfinite(candidate_distance) and candidate_distance > max_distance:
                continue

            z_hat = predict_range_bearing(mu, landmark_position)
            innovation = innovation_range_bearing(z, z_hat)
            landmark_index = None if state_index is None else get_landmark_state_index(state_index, track.track_id)
            if landmark_index is None:
                H = np.zeros((2, len(mu)), dtype=float)
                H[:, :3] = range_bearing_jacobian_pose(mu, landmark_position)
            else:
                H = range_bearing_jacobian_full_state(mu, landmark_index)
            R = measurement_noise_matrix(measurement_config)
            innovation_covariance = H @ Sigma @ H.T + R
            mahalanobis_distance = compute_mahalanobis_distance(innovation, innovation_covariance)

            candidate = AssociationCandidate(
                observation_index=observation_index,
                track_id=track.track_id,
                innovation=innovation,
                distance=candidate_distance,
                mahalanobis_distance=mahalanobis_distance,
            )
            if best_candidate is None or candidate.mahalanobis_distance < best_candidate.mahalanobis_distance:
                second_best_candidate = best_candidate
                best_candidate = candidate
            elif second_best_candidate is None or candidate.mahalanobis_distance < second_best_candidate.mahalanobis_distance:
                second_best_candidate = candidate

        if best_candidate is None:
            unmatched_observations.append(observation_index)
            continue

        if is_association_ambiguous(
            best_score=best_candidate.mahalanobis_distance,
            second_best_score=None if second_best_candidate is None else second_best_candidate.mahalanobis_distance,
            ambiguity_ratio_threshold=ambiguity_ratio_threshold,
            ambiguity_margin_threshold=ambiguity_margin_threshold,
        ):
            ambiguous.append(best_candidate)
            unmatched_observations.append(observation_index)
            continue

        if best_candidate.mahalanobis_distance <= mahalanobis_threshold:
            matched.append(
                AssociationMatch(
                    observation_index=best_candidate.observation_index,
                    track_id=best_candidate.track_id,
                    innovation=np.array(best_candidate.innovation, dtype=float, copy=True),
                    distance=best_candidate.distance,
                    mahalanobis_distance=best_candidate.mahalanobis_distance,
                )
            )
            used_track_ids.add(best_candidate.track_id)
        else:
            rejected.append(best_candidate)
            unmatched_observations.append(observation_index)

    return AssociationResult(
        matched=tuple(matched),
        unmatched_observations=tuple(unmatched_observations),
        rejected=tuple(rejected),
        ambiguous=tuple(ambiguous),
        method="mahalanobis",
        gating_applied=True,
    )


def joseph_covariance_update(Sigma: np.ndarray, K: np.ndarray, H: np.ndarray, R: np.ndarray):
    identity = np.eye(Sigma.shape[0], dtype=float)
    updated = (identity - K @ H) @ Sigma @ (identity - K @ H).T + K @ R @ K.T
    updated = symmetrize_covariance(updated)
    return ensure_positive_semidefinite(updated)


def ekf_update_pose_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    observation: LandmarkObservation,
    landmark_position: np.ndarray,
    measurement_config: MeasurementModelConfig,
):
    z = observation_to_measurement_vector(observation)
    z_hat = predict_range_bearing(mu, landmark_position)
    H = range_bearing_jacobian_pose(mu, landmark_position)
    R = measurement_noise_matrix(measurement_config)

    innovation = innovation_range_bearing(z, z_hat)
    S = H @ Sigma @ H.T + R
    S = symmetrize_covariance(S)
    S_inv = np.linalg.inv(S)
    K = Sigma @ H.T @ S_inv

    updated_mu = np.array(mu, dtype=float, copy=True)
    updated_mu = updated_mu + K @ innovation
    updated_mu = normalize_state_angle(updated_mu)

    updated_Sigma = joseph_covariance_update(Sigma, K, H, R)
    nis = float(innovation.T @ S_inv @ innovation)
    return EkfUpdateResult(
        mu=updated_mu,
        Sigma=updated_Sigma,
        innovation=innovation,
        kalman_gain=K,
        nis=nis,
    )


def ekf_update_pose_only_batch(
    mu: np.ndarray,
    Sigma: np.ndarray,
    observations: Sequence[LandmarkObservation],
    landmark_positions: Sequence[np.ndarray],
    measurement_config: MeasurementModelConfig,
):
    if len(observations) != len(landmark_positions):
        raise ValueError("Observations and landmark_positions must have the same length.")

    updated_mu = np.array(mu, dtype=float, copy=True)
    updated_Sigma = np.array(Sigma, dtype=float, copy=True)
    update_results: list[EkfUpdateResult] = []

    for observation, landmark_position in zip(observations, landmark_positions):
        update_result = ekf_update_pose_only(
            updated_mu,
            updated_Sigma,
            observation,
            landmark_position,
            measurement_config,
        )
        updated_mu = update_result.mu
        updated_Sigma = update_result.Sigma
        update_results.append(update_result)

    return updated_mu, updated_Sigma, update_results


def ekf_update_pose_only_batch_gated(
    mu: np.ndarray,
    Sigma: np.ndarray,
    observations: Sequence[LandmarkObservation],
    landmark_positions: Sequence[np.ndarray],
    measurement_config: MeasurementModelConfig,
    nis_threshold: float | None,
):
    if len(observations) != len(landmark_positions):
        raise ValueError("Observations and landmark_positions must have the same length.")

    updated_mu = np.array(mu, dtype=float, copy=True)
    updated_Sigma = np.array(Sigma, dtype=float, copy=True)
    update_results: list[EkfUpdateResult] = []
    rejected_count = 0

    for observation, landmark_position in zip(observations, landmark_positions):
        update_result = ekf_update_pose_only(
            updated_mu,
            updated_Sigma,
            observation,
            landmark_position,
            measurement_config,
        )
        if not is_nis_accepted(update_result.nis, nis_threshold):
            rejected_count += 1
            continue

        updated_mu = update_result.mu
        updated_Sigma = update_result.Sigma
        update_results.append(update_result)

    return updated_mu, updated_Sigma, update_results, rejected_count


def predict_landmark_from_observation(mu_pose: np.ndarray, observation: LandmarkObservation):
    heading = float(mu_pose[2] + observation.bearing)
    return np.array(
        [
            mu_pose[0] + observation.range * np.cos(heading),
            mu_pose[1] + observation.range * np.sin(heading),
        ],
        dtype=float,
    )


def range_bearing_jacobian_landmark(mu: np.ndarray, landmark_position: np.ndarray):
    dx = float(landmark_position[0] - mu[0])
    dy = float(landmark_position[1] - mu[1])
    squared_range = dx * dx + dy * dy
    if squared_range < 1e-12:
        raise ValueError("Landmark is too close to the robot pose for a stable landmark Jacobian.")

    predicted_range = float(np.sqrt(squared_range))
    return np.array(
        [
            [dx / predicted_range, dy / predicted_range],
            [-dy / squared_range, dx / squared_range],
        ],
        dtype=float,
    )


def range_bearing_jacobian_full_state(mu: np.ndarray, landmark_index: int):
    if landmark_index < 3 or landmark_index + 1 >= len(mu):
        raise IndexError("landmark_index must point to the first element of a 2D landmark in the EKF state.")

    landmark_position = np.asarray(mu[landmark_index:landmark_index + 2], dtype=float)
    H = np.zeros((2, len(mu)), dtype=float)
    H[:, :3] = range_bearing_jacobian_pose(mu, landmark_position)
    H[:, landmark_index:landmark_index + 2] = range_bearing_jacobian_landmark(mu, landmark_position)
    return H


def augment_state_with_landmark(
    mu: np.ndarray,
    Sigma: np.ndarray,
    observation: LandmarkObservation,
    measurement_config: MeasurementModelConfig,
):
    pose = np.asarray(mu[:3], dtype=float)
    landmark_position = predict_landmark_from_observation(pose, observation)
    heading = float(pose[2] + observation.bearing)

    J_pose = np.array(
        [
            [1.0, 0.0, -observation.range * np.sin(heading)],
            [0.0, 1.0, observation.range * np.cos(heading)],
        ],
        dtype=float,
    )
    J_measurement = np.array(
        [
            [np.cos(heading), -observation.range * np.sin(heading)],
            [np.sin(heading), observation.range * np.cos(heading)],
        ],
        dtype=float,
    )

    R = measurement_noise_matrix(measurement_config)
    state_size = len(mu)
    augmented_mu = np.concatenate([np.asarray(mu, dtype=float), landmark_position])

    augmented_Sigma = np.zeros((state_size + 2, state_size + 2), dtype=float)
    augmented_Sigma[:state_size, :state_size] = Sigma

    pose_cross_covariance = np.asarray(Sigma[:3, :], dtype=float)
    landmark_cross_covariance = J_pose @ pose_cross_covariance
    landmark_covariance = J_pose @ Sigma[:3, :3] @ J_pose.T + J_measurement @ R @ J_measurement.T

    augmented_Sigma[state_size:, :state_size] = landmark_cross_covariance
    augmented_Sigma[:state_size, state_size:] = landmark_cross_covariance.T
    augmented_Sigma[state_size:, state_size:] = landmark_covariance
    augmented_Sigma = ensure_positive_semidefinite(augmented_Sigma)
    return normalize_state_angle(augmented_mu), augmented_Sigma


def ekf_update_full_state(
    mu: np.ndarray,
    Sigma: np.ndarray,
    observation: LandmarkObservation,
    landmark_index: int,
    measurement_config: MeasurementModelConfig,
):
    landmark_position = np.asarray(mu[landmark_index:landmark_index + 2], dtype=float)
    z = observation_to_measurement_vector(observation)
    z_hat = predict_range_bearing(mu, landmark_position)
    H = range_bearing_jacobian_full_state(mu, landmark_index)
    R = measurement_noise_matrix(measurement_config)

    innovation = innovation_range_bearing(z, z_hat)
    S = H @ Sigma @ H.T + R
    S = symmetrize_covariance(S)
    S_inv = np.linalg.inv(S)
    K = Sigma @ H.T @ S_inv

    updated_mu = np.array(mu, dtype=float, copy=True)
    updated_mu = updated_mu + K @ innovation
    updated_mu = normalize_state_angle(updated_mu)

    updated_Sigma = joseph_covariance_update(Sigma, K, H, R)
    nis = float(innovation.T @ S_inv @ innovation)
    return EkfUpdateResult(
        mu=updated_mu,
        Sigma=updated_Sigma,
        innovation=innovation,
        kalman_gain=K,
        nis=nis,
    )


def extract_truth_landmark_positions(walls: Sequence[WallSegment], merge_tolerance: float = 1e-6):
    truth_landmarks: list[np.ndarray] = []

    for wall in walls:
        for point in (wall.start, wall.end):
            point_array = np.array(point, dtype=float)
            if any(np.linalg.norm(point_array - existing) <= merge_tolerance for existing in truth_landmarks):
                continue
            truth_landmarks.append(point_array)

    return truth_landmarks


def simulate_landmark_observations_from_truth(
    pose: np.ndarray,
    landmark_positions: Sequence[np.ndarray],
    measurement_config: MeasurementModelConfig,
    sensor_config: SensorConfig,
    rng,
    max_range: float,
    max_observations: int,
):
    if measurement_config.model_type != "range_bearing":
        raise ValueError(f"Unsupported measurement model type: {measurement_config.model_type}")

    if max_observations <= 0:
        return TruthObservationSet(observations=[], landmark_positions=[])

    half_fov = np.radians(sensor_config.fov_degrees / 2.0)
    visible_observations: list[tuple[float, LandmarkObservation]] = []

    for source_id, landmark_position in enumerate(landmark_positions):
        predicted_measurement = predict_range_bearing(pose, landmark_position)
        predicted_range = float(predicted_measurement[0])
        predicted_bearing = float(predicted_measurement[1])

        if predicted_range > max_range or abs(predicted_bearing) > half_fov:
            continue

        noisy_range = max(0.0, rng.normal(predicted_range, measurement_config.range_std))
        noisy_bearing = normalize_angle(rng.normal(predicted_bearing, np.radians(measurement_config.bearing_std_deg)))
        observation = LandmarkObservation(
            range=noisy_range,
            bearing=noisy_bearing,
            world_position=np.array(landmark_position, dtype=float),
            source_id=source_id,
        )
        visible_observations.append((predicted_range, observation))

    visible_observations.sort(key=lambda item: item[0])
    selected_observations = [item[1] for item in visible_observations[:max_observations]]
    selected_positions = [np.array(observation.world_position, dtype=float) for observation in selected_observations]
    return TruthObservationSet(observations=selected_observations, landmark_positions=selected_positions)


def get_intersection(ray_origin, ray_dir, wall: WallSegment):
    p = np.array(ray_origin, dtype=float)
    d = np.array(ray_dir, dtype=float)
    a = np.array(wall.start, dtype=float)
    b = np.array(wall.end, dtype=float)

    v1 = p - a
    v2 = b - a
    v3 = np.array([-d[1], d[0]], dtype=float)

    dot = np.dot(v2, v3)
    if abs(dot) < 1e-6:
        return None, float("inf")

    t1 = cross2d(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot

    if t1 >= 0 and 0 <= t2 <= 1:
        hit_point = p + t1 * d
        return hit_point, t1

    return None, float("inf")


def apply_sensor_noise(distance: float, sensor_config: SensorConfig, rng):
    noise_config = sensor_config.noise
    if not noise_config.enabled:
        return distance

    distance_ratio = np.clip(distance / sensor_config.max_range, 0.0, 1.0)
    relative_std = np.interp(
        distance_ratio,
        [0.0, 1.0],
        [noise_config.min_relative_std, noise_config.max_relative_std],
    )
    noisy_distance = rng.normal(distance, distance * relative_std)
    return float(np.clip(noisy_distance, 0.0, sensor_config.max_range))


def simulate_lidar(
    x: float,
    y: float,
    theta: float,
    walls: Sequence[WallSegment],
    sensor_config: SensorConfig,
    rng,
    forward_sector_rad: float = np.radians(10.0),
):
    angles = np.linspace(
        -np.radians(sensor_config.fov_degrees / 2),
        np.radians(sensor_config.fov_degrees / 2),
        sensor_config.num_rays,
    )
    measurements: list[ScanMeasurement] = []
    scan_samples: list[ScanMeasurement] = []
    min_dist_forward = float("inf")

    for angle in angles:
        ray_theta = theta + angle
        ray_dir = [np.cos(ray_theta), np.sin(ray_theta)]

        min_dist = sensor_config.max_range
        for wall in walls:
            hit_point, dist = get_intersection((x, y), ray_dir, wall)
            if hit_point is not None and dist < min_dist:
                min_dist = dist

        if min_dist < sensor_config.max_range:
            noisy_dist = apply_sensor_noise(min_dist, sensor_config, rng)
            measurement = ScanMeasurement(angle=float(angle), distance=noisy_dist)
            measurements.append(measurement)
            scan_samples.append(measurement)
        else:
            scan_samples.append(ScanMeasurement(angle=float(angle), distance=sensor_config.max_range))

        if abs(angle) < forward_sector_rad:
            min_dist_forward = min(min_dist_forward, min_dist)

    return LidarScan(
        measurements=measurements,
        scan_samples=scan_samples,
        min_dist_forward=float(min_dist_forward),
    )


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


def suppress_nearby_corners(candidates, nms_radius: int):
    selected = []
    blocked_indices = set()

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        index = candidate["index"]
        if index in blocked_indices:
            continue
        selected.append(candidate)
        blocked_indices.update(range(index - nms_radius, index + nms_radius + 1))

    return sorted(selected, key=lambda item: item["index"])


def extract_landmarks(measurements: Sequence[ScanMeasurement], config: FeatureExtractionConfig):
    window_size = config.window_size
    if len(measurements) < (2 * window_size + 1):
        return []

    local_points = [
        np.array([measurement.distance * np.cos(measurement.angle), measurement.distance * np.sin(measurement.angle)], dtype=float)
        for measurement in measurements
    ]

    candidates = []
    for index in range(window_size, len(local_points) - window_size):
        left_points = local_points[index - window_size:index]
        right_points = local_points[index + 1:index + 1 + window_size]
        candidate_point = local_points[index]

        left_chain = left_points + [candidate_point]
        right_chain = [candidate_point] + right_points
        if any(np.linalg.norm(b - a) > config.max_neighbor_gap for a, b in zip(left_chain, left_chain[1:])):
            continue
        if any(np.linalg.norm(b - a) > config.max_neighbor_gap for a, b in zip(right_chain, right_chain[1:])):
            continue

        left_span = np.linalg.norm(left_points[-1] - left_points[0])
        right_span = np.linalg.norm(right_points[-1] - right_points[0])
        if left_span < config.min_segment_span or right_span < config.min_segment_span:
            continue

        left_center, left_direction = fit_line_direction(left_points)
        right_center, right_direction = fit_line_direction(right_points)

        line_alignment = np.clip(abs(np.dot(left_direction, right_direction)), 0.0, 1.0)
        corner_angle_deg = np.degrees(np.arccos(line_alignment))
        if not (config.min_corner_angle_deg <= corner_angle_deg <= config.max_corner_angle_deg):
            continue

        left_residual = point_line_distance(candidate_point, left_center, left_direction)
        right_residual = point_line_distance(candidate_point, right_center, right_direction)
        if left_residual > config.max_line_residual or right_residual > config.max_line_residual:
            continue

        candidates.append({"index": index, "score": corner_angle_deg})

    filtered_candidates = suppress_nearby_corners(candidates, config.nms_radius)
    return [measurements[candidate["index"]] for candidate in filtered_candidates]


def transform_measurements(measurements: Sequence[ScanMeasurement], pose: np.ndarray):
    points_x, points_y = [], []
    for measurement in measurements:
        world_theta = pose[2] + measurement.angle
        points_x.append(pose[0] + measurement.distance * np.cos(world_theta))
        points_y.append(pose[1] + measurement.distance * np.sin(world_theta))
    return points_x, points_y


def measurements_to_world_points(measurements: Sequence[ScanMeasurement], pose: np.ndarray):
    world_points = []
    for measurement in measurements:
        world_theta = pose[2] + measurement.angle
        world_points.append(
            np.array(
                [
                    pose[0] + measurement.distance * np.cos(world_theta),
                    pose[1] + measurement.distance * np.sin(world_theta),
                ],
                dtype=float,
            )
        )
    return world_points


def update_persistent_landmarks(
    world_points: Sequence[np.ndarray],
    persistent_landmarks: Sequence[PersistentLandmark],
    feature_config: FeatureExtractionConfig,
):
    updated_landmarks: list[PersistentLandmark] = []
    for landmark in persistent_landmarks:
        remaining = landmark.ttl - 1
        if remaining > 0:
            updated_landmarks.append(PersistentLandmark(position=landmark.position, ttl=remaining))

    for point in world_points:
        best_match = None
        best_distance = feature_config.association_radius
        for landmark in updated_landmarks:
            distance = np.linalg.norm(point - landmark.position)
            if distance <= best_distance:
                best_distance = distance
                best_match = landmark

        if best_match is None:
            updated_landmarks.append(PersistentLandmark(position=point, ttl=feature_config.persistence_frames))
        else:
            best_match.position = 0.5 * (best_match.position + point)
            best_match.ttl = feature_config.persistence_frames

    return updated_landmarks


def initialize_landmark_track_state():
    return LandmarkTrackState(tracks={})


def _compute_track_quality(observation_count: int, staleness_frames: int, ttl: int):
    confidence = min(1.0, observation_count / 5.0)
    freshness = max(0.0, 1.0 - (staleness_frames / max(ttl, 1)))
    return float(confidence * freshness)


def create_landmark_track(point: np.ndarray, frame_index: int, feature_config: FeatureExtractionConfig, track_id: int):
    ttl = feature_config.persistence_frames
    covariance = np.eye(2, dtype=float) * (feature_config.association_radius ** 2)
    return LandmarkTrack(
        track_id=track_id,
        position=np.array(point, dtype=float, copy=True),
        covariance=covariance,
        observation_count=1,
        last_seen_frame=frame_index,
        ttl=ttl,
        quality_score=_compute_track_quality(1, 0, ttl),
    )


def update_landmark_track(
    track: LandmarkTrack,
    point: np.ndarray,
    frame_index: int,
    feature_config: FeatureExtractionConfig,
):
    previous_count = max(track.observation_count, 1)
    blend = 1.0 / (previous_count + 1.0)
    track.position = (1.0 - blend) * track.position + blend * np.asarray(point, dtype=float)
    track.observation_count = previous_count + 1
    track.last_seen_frame = frame_index
    track.ttl = feature_config.persistence_frames
    track.quality_score = _compute_track_quality(track.observation_count, 0, track.ttl)

    if track.covariance is not None:
        base_variance = feature_config.association_radius ** 2
        confidence_scale = max(0.2, 1.0 / track.observation_count)
        track.covariance = np.eye(2, dtype=float) * (base_variance * confidence_scale)

    return track


def prune_landmark_tracks(
    track_state: LandmarkTrackState,
    frame_index: int,
    protected_track_ids: Sequence[int] = (),
):
    protected_track_ids = set(protected_track_ids)
    expired_track_ids: list[int] = []
    for track_id, track in track_state.tracks.items():
        staleness_frames = max(0, frame_index - track.last_seen_frame)
        if staleness_frames > track.ttl and track_id not in protected_track_ids:
            expired_track_ids.append(track_id)
            continue
        track.quality_score = _compute_track_quality(track.observation_count, staleness_frames, track.ttl)

    for track_id in expired_track_ids:
        del track_state.tracks[track_id]

    return track_state


def update_landmark_track_state(
    world_points: Sequence[np.ndarray],
    association_result: AssociationResult,
    track_state: LandmarkTrackState,
    frame_index: int,
    feature_config: FeatureExtractionConfig,
    protected_track_ids: Sequence[int] = (),
):
    prune_landmark_tracks(track_state, frame_index, protected_track_ids)
    assignments: list[TrackAssignment] = []
    matched_observation_indices: set[int] = set()
    used_track_ids: set[int] = set()

    for match in association_result.matched:
        track = track_state.tracks.get(match.track_id)
        if track is None or match.observation_index >= len(world_points):
            continue

        point_array = np.asarray(world_points[match.observation_index], dtype=float)
        update_landmark_track(track, point_array, frame_index, feature_config)
        assignments.append(TrackAssignment(observation_index=match.observation_index, track_id=track.track_id, created=False))
        matched_observation_indices.add(match.observation_index)
        used_track_ids.add(track.track_id)

    for observation_index, point in enumerate(world_points):
        if observation_index in matched_observation_indices:
            continue

        point_array = np.asarray(point, dtype=float)
        best_track = None
        best_distance = feature_config.association_radius
        for track in track_state.tracks.values():
            if track.track_id in protected_track_ids or track.track_id in used_track_ids:
                continue
            distance = float(np.linalg.norm(point_array - track.position))
            if distance <= best_distance:
                best_distance = distance
                best_track = track

        if best_track is not None:
            update_landmark_track(best_track, point_array, frame_index, feature_config)
            assignments.append(TrackAssignment(observation_index=observation_index, track_id=best_track.track_id, created=False))
            used_track_ids.add(best_track.track_id)
        else:
            track_id = track_state.next_track_id
            track_state.tracks[track_id] = create_landmark_track(point_array, frame_index, feature_config, track_id)
            track_state.next_track_id += 1
            assignments.append(TrackAssignment(observation_index=observation_index, track_id=track_id, created=True))
            used_track_ids.add(track_id)

    return TrackUpdateResult(track_state=track_state, assignments=tuple(assignments))


def sync_landmark_tracks_with_state(
    track_state: LandmarkTrackState,
    state_index: EkfSlamStateIndex,
    mu: np.ndarray,
    Sigma: np.ndarray,
):
    for track_id, landmark_index in state_index.track_id_to_index.items():
        track = track_state.tracks.get(track_id)
        if track is None or landmark_index + 1 >= len(mu):
            continue
        track.position = np.array(mu[landmark_index:landmark_index + 2], dtype=float, copy=True)
        track.covariance = np.array(
            Sigma[landmark_index:landmark_index + 2, landmark_index:landmark_index + 2],
            dtype=float,
            copy=True,
        )

    return track_state


def get_voxel_key(point_x: float, point_y: float, voxel_size: float):
    return (int(point_x // voxel_size), int(point_y // voxel_size))


def _range_relative_std(distance: float, sensor_config: SensorConfig):
    noise_config = sensor_config.noise
    distance_ratio = np.clip(distance / sensor_config.max_range, 0.0, 1.0)
    return float(
        np.interp(
            distance_ratio,
            [0.0, 1.0],
            [noise_config.min_relative_std, noise_config.max_relative_std],
        )
    )


def _measurement_weight(distance: float, sensor_config: SensorConfig, voxel_config: VoxelGridConfig):
    epsilon = 1e-6
    if voxel_config.weighting_mode == "inverse_variance" and sensor_config.noise.enabled:
        sigma = max(distance * _range_relative_std(distance, sensor_config), epsilon)
        return 1.0 / (sigma ** 2)

    safe_distance = max(distance, epsilon)
    return 1.0 / (safe_distance ** voxel_config.distance_weight_power)


def _advance_voxel_state(state: VoxelCellState, frame_index: int, voxel_config: VoxelGridConfig):
    if frame_index <= state.last_frame:
        return

    decay_factor = voxel_config.temporal_decay ** (frame_index - state.last_frame)
    state.sum_w *= decay_factor
    state.sum_wx *= decay_factor
    state.sum_wy *= decay_factor
    state.count *= decay_factor
    state.best_weight *= decay_factor
    state.last_frame = frame_index

    if state.best_weight < 1e-12:
        state.best_distance = float("inf")


def update_voxel_grid(
    points_x,
    points_y,
    measurements: Sequence[ScanMeasurement],
    voxel_state,
    voxel_config: VoxelGridConfig,
    sensor_config: SensorConfig,
    frame_index: int,
):
    for point_x, point_y, measurement in zip(points_x, points_y, measurements):
        voxel_key = get_voxel_key(point_x, point_y, voxel_config.voxel_size)
        state = voxel_state[voxel_key]
        _advance_voxel_state(state, frame_index, voxel_config)

        weight = _measurement_weight(measurement.distance, sensor_config, voxel_config)
        state.sum_w += weight
        state.sum_wx += weight * point_x
        state.sum_wy += weight * point_y
        state.count += 1.0

        if (
            voxel_config.best_observation_override
            and measurement.distance <= voxel_config.best_observation_max_distance
            and weight >= state.best_weight
        ):
            state.best_x = point_x
            state.best_y = point_y
            state.best_weight = weight
            state.best_distance = measurement.distance

    averaged_points_x, averaged_points_y = [], []
    for state in voxel_state.values():
        _advance_voxel_state(state, frame_index, voxel_config)
        if state.count >= voxel_config.min_points_per_voxel and state.sum_w > 1e-12:
            if (
                voxel_config.best_observation_override
                and state.best_weight > 1e-12
                and state.best_distance <= voxel_config.best_observation_max_distance
            ):
                averaged_points_x.append(state.best_x)
                averaged_points_y.append(state.best_y)
            else:
                averaged_points_x.append(state.sum_wx / state.sum_w)
                averaged_points_y.append(state.sum_wy / state.sum_w)

    return averaged_points_x, averaged_points_y


def ekf_predict(mu, Sigma, measured_distance: float, measured_turn: float, noise_config: OdometryNoiseConfig):
    x, y, theta = mu[0], mu[1], mu[2]

    theta_new = normalize_angle(theta + measured_turn)

    x_new = x + measured_distance * np.cos(theta_new)
    y_new = y + measured_distance * np.sin(theta_new)

    mu[0], mu[1], mu[2] = x_new, y_new, theta_new
    mu = normalize_state_angle(mu)

    state_size = len(mu)
    G = np.eye(state_size)
    G[0, 2] = -measured_distance * np.sin(theta_new)
    G[1, 2] = measured_distance * np.cos(theta_new)

    var_dist = noise_config.distance_std ** 2
    var_turn = np.radians(noise_config.angle_std_deg) ** 2
    q_control = np.array([[var_dist, 0], [0, var_turn]])

    v = np.array(
        [
            [np.cos(theta_new), -measured_distance * np.sin(theta_new)],
            [np.sin(theta_new), measured_distance * np.cos(theta_new)],
            [0, 1],
        ]
    )

    r_pose = v @ q_control @ v.T
    r = np.zeros((state_size, state_size))
    r[0:3, 0:3] = r_pose

    Sigma = G @ Sigma @ G.T + r
    Sigma = symmetrize_covariance(Sigma)
    Sigma = ensure_positive_semidefinite(Sigma)
    return mu, Sigma
