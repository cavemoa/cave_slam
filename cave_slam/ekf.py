from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .slam import (
    AssociationMatch,
    AssociationResult,
    EkfUpdateResult,
    LandmarkObservation,
    LandmarkTrackState,
    TruthObservationSet,
    TrackUpdateResult,
    association_confidence_key,
    associate_landmarks_mahalanobis,
    associate_landmarks_nearest_neighbor,
    augment_state_with_landmark,
    ekf_update_full_state,
    ekf_update_pose_only_batch_gated,
    get_landmark_state_index,
    is_ekf_compatible_landmark_type,
    is_nis_accepted,
    is_track_ready_for_augmentation,
    landmark_type_priority,
    type_aware_track_quality,
)

if TYPE_CHECKING:
    from .sim import SimulationState


def associate_feature_observations(
    state: SimulationState,
    feature_observations: Sequence[LandmarkObservation],
):
    association_config = state.config.ekf.association
    track_state = state.slam_state.landmark_track_state
    if association_config.method == "nearest_neighbor":
        return associate_landmarks_nearest_neighbor(
            feature_observations,
            track_state,
            association_config,
            max_distance=association_config.max_distance,
            min_track_quality=association_config.min_track_quality,
            ambiguity_ratio_threshold=association_config.ambiguity_ratio_threshold,
            ambiguity_margin_threshold=association_config.ambiguity_margin_threshold,
            mu=state.slam_state.mu,
            state_index=state.slam_state.ekf_slam_index,
        )

    return associate_landmarks_mahalanobis(
        feature_observations,
        track_state,
        association_config,
        state.slam_state.mu,
        state.slam_state.Sigma,
        state.config.ekf.measurement,
        max_distance=association_config.max_distance,
        mahalanobis_threshold=association_config.mahalanobis_threshold,
        min_track_quality=association_config.min_track_quality,
        ambiguity_ratio_threshold=association_config.ambiguity_ratio_threshold,
        ambiguity_margin_threshold=association_config.ambiguity_margin_threshold,
        state_index=state.slam_state.ekf_slam_index,
    )


def extract_associated_track_positions(
    track_state: LandmarkTrackState,
    matches: Sequence[AssociationMatch],
):
    associated_positions: list[np.ndarray] = []
    for match in matches:
        track = track_state.tracks.get(match.track_id)
        if track is None:
            continue
        associated_positions.append(np.array(track.position, dtype=float, copy=True))
    return associated_positions


def prioritize_association_matches(
    association_result: AssociationResult,
    observations: Sequence[LandmarkObservation],
):
    return tuple(
        sorted(
            association_result.matched,
            key=lambda match: (
                landmark_type_priority(observations[match.observation_index].landmark_type),
                association_confidence_key(match, association_result.method),
                -float(np.clip(observations[match.observation_index].confidence, 0.0, 1.0)),
            ),
        )
    )


def prioritize_track_assignments(
    assignments: Sequence,
    track_state: LandmarkTrackState,
    observations: Sequence[LandmarkObservation],
):
    return tuple(
        sorted(
            assignments,
            key=lambda assignment: (
                landmark_type_priority(observations[assignment.observation_index].landmark_type),
                -type_aware_track_quality(track_state.tracks[assignment.track_id]),
                -track_state.tracks[assignment.track_id].observation_count,
            )
            if assignment.track_id in track_state.tracks and assignment.observation_index < len(observations)
            else (99, float("inf"), float("inf")),
        )
    )


def _register_augmented_landmark(state: SimulationState, track_id: int):
    landmark_index = len(state.slam_state.mu) - 2
    state.slam_state.ekf_slam_index.track_id_to_index[track_id] = landmark_index
    if track_id not in state.slam_state.ekf_slam_index.state_track_ids:
        state.slam_state.ekf_slam_index.state_track_ids.append(track_id)
    return landmark_index


def apply_full_slam_correction(
    state: SimulationState,
    feature_observations: Sequence[LandmarkObservation],
    association_result: AssociationResult,
    track_update_result: TrackUpdateResult,
):
    pose_update_config = state.config.ekf.pose_update
    if not pose_update_config.enabled:
        return [], (), 0

    max_updates = max(0, pose_update_config.max_updates_per_frame)
    if max_updates == 0:
        return [], (), 0
    update_results: list[EkfUpdateResult] = []
    augmented_track_ids: list[int] = []
    rejected_count = 0

    prioritized_matches = prioritize_association_matches(association_result, feature_observations)
    for match in prioritized_matches[:max_updates]:
        landmark_index = get_landmark_state_index(state.slam_state.ekf_slam_index, match.track_id)
        if landmark_index is None:
            continue
        observation = feature_observations[match.observation_index]
        if not is_ekf_compatible_landmark_type(observation.landmark_type):
            continue

        update_result = ekf_update_full_state(
            state.slam_state.mu,
            state.slam_state.Sigma,
            observation,
            landmark_index,
            state.config.ekf.measurement,
        )
        if not is_nis_accepted(update_result.nis, pose_update_config.nis_threshold):
            rejected_count += 1
            continue
        state.slam_state.mu = update_result.mu
        state.slam_state.Sigma = update_result.Sigma
        update_results.append(update_result)

    prioritized_assignments = prioritize_track_assignments(
        track_update_result.assignments,
        state.slam_state.landmark_track_state,
        feature_observations,
    )
    for assignment in prioritized_assignments:
        if get_landmark_state_index(state.slam_state.ekf_slam_index, assignment.track_id) is not None:
            continue
        if assignment.observation_index >= len(feature_observations):
            continue
        observation = feature_observations[assignment.observation_index]
        if not is_ekf_compatible_landmark_type(observation.landmark_type):
            continue
        track = state.slam_state.landmark_track_state.tracks.get(assignment.track_id)
        if track is None or not is_track_ready_for_augmentation(track, state.config.ekf.augmentation):
            continue

        state.slam_state.mu, state.slam_state.Sigma = augment_state_with_landmark(
            state.slam_state.mu,
            state.slam_state.Sigma,
            observation,
            state.config.ekf.measurement,
        )
        _register_augmented_landmark(state, assignment.track_id)
        augmented_track_ids.append(assignment.track_id)

    return update_results, tuple(augmented_track_ids), rejected_count


def apply_pose_only_ekf_correction(
    state: SimulationState,
    truth_observation_set: TruthObservationSet | None,
    feature_observations: Sequence[LandmarkObservation],
    association_result: AssociationResult,
):
    pose_update_config = state.config.ekf.pose_update
    if not pose_update_config.enabled:
        return [], 0

    max_updates = max(0, pose_update_config.max_updates_per_frame)
    if max_updates == 0:
        return [], 0

    if pose_update_config.use_truth_observations:
        if truth_observation_set is None:
            return [], 0
        selected_observations = truth_observation_set.observations[:max_updates]
        selected_landmarks = truth_observation_set.landmark_positions[:max_updates]
    else:
        prioritized_matches = prioritize_association_matches(association_result, feature_observations)
        if not prioritized_matches:
            return [], 0
        selected_matches = [
            match
            for match in prioritized_matches
            if is_ekf_compatible_landmark_type(feature_observations[match.observation_index].landmark_type)
        ][:max_updates]
        if not selected_matches:
            return [], 0
        selected_observations = [feature_observations[match.observation_index] for match in selected_matches]
        selected_landmarks = extract_associated_track_positions(
            state.slam_state.landmark_track_state,
            selected_matches,
        )

    if not selected_observations or len(selected_observations) != len(selected_landmarks):
        return [], 0

    updated_mu, updated_Sigma, update_results, rejected_count = ekf_update_pose_only_batch_gated(
        state.slam_state.mu,
        state.slam_state.Sigma,
        selected_observations,
        selected_landmarks,
        state.config.ekf.measurement,
        pose_update_config.nis_threshold,
    )
    state.slam_state.mu = updated_mu
    state.slam_state.Sigma = updated_Sigma
    return update_results, rejected_count
