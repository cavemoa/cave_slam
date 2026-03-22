# EKF Roadmap

This document describes a staged plan for improving the EKF functionality in the `cave_slam` project. It is intended to be used as a stable reference while developing the estimator over time.

The roadmap is designed to be:

- incremental
- numerically safer
- easy to validate at each stage
- compatible with the current `cave_slam` package structure

The current state of the project is that the EKF is still prediction-only. The simulation already computes:

- noisy odometry
- a predicted pose estimate
- lidar-derived feature observations
- persistent visual landmark-like detections

However, the estimator does not yet perform a measurement correction step.

## Stage Names

Use these exact phrases when referring to progress:

- `Stage 0: EKF Predict Hardening`
- `Stage 1: Pose-Only Measurement Model`
- `Stage 2: Truth Landmark Update Harness`
- `Stage 3: Landmark Track Layer`
- `Stage 4: Data Association And Gating`
- `Stage 5: Pose-Only EKF Correction`
- `Stage 6: EKF Diagnostics`
- `Stage 7: Full EKF-SLAM Augmentation`
- `Stage 8: Robustification`

If you want to refer to a specific deliverable using a shorter phrase, use:

- `Predict Hardening`
- `Measurement Model`
- `Truth Harness`
- `Track Layer`
- `Association`
- `Pose Correction`
- `Diagnostics`
- `State Augmentation`
- `Robustification`

These phrases should be treated as agreed keywords for future discussion.

## Current EKF Situation

At the moment:

- the EKF state is just the robot pose
- the project uses `ekf_predict(...)`
- there is no EKF measurement update
- `persistent_landmarks` are useful for visualization, but they are not yet an estimator map
- there is no explicit data association, innovation, gating, or landmark state augmentation

This means the roadmap should start with safer prediction and a minimal correction path before attempting full EKF-SLAM.

## Stage 0: EKF Predict Hardening

### Goal

Make the existing prediction step numerically safer and easier to debug before any correction logic is added.

### Why This Stage Comes First

If the predict step is unstable, a later update step will be much harder to trust and debug.

### Add In `cave_slam/slam.py`

- `normalize_angle(angle: float) -> float`
- `normalize_state_angle(mu: np.ndarray) -> np.ndarray`
- `symmetrize_covariance(Sigma: np.ndarray) -> np.ndarray`
- optionally `ensure_positive_semidefinite(Sigma: np.ndarray, jitter: float = ...) -> np.ndarray`

### Update Existing Functions

- refactor `ekf_predict(...)` to use `normalize_angle`
- symmetrize `Sigma` after prediction
- optionally clamp or regularize the diagonal if required

### Optional Dataclass

- `EkfDebugInfo`
  - `trace_sigma`
  - `pose_std_x`
  - `pose_std_y`
  - `pose_std_theta`

### Add In `cave_slam/sim.py`

- optionally include EKF debug fields in `StepResult`

### Agreed Keyword

- `Predict Hardening`

## Stage 1: Pose-Only Measurement Model

### Goal

Add the measurement math needed for correction while keeping the EKF state as pose only.

### Why This Stage Matters

It lets the project validate the correction equations without the complexity of dynamic landmark state augmentation.

### Add In `cave_slam/slam.py`

- `MeasurementModelConfig`
  - `range_std`
  - `bearing_std_deg`
  - `model_type`
- `LandmarkObservation`
  - `range`
  - `bearing`
  - `world_position` optional for debugging
  - `source_id` optional
- `predict_range_bearing(mu: np.ndarray, landmark_position: np.ndarray) -> np.ndarray`
- `range_bearing_jacobian_pose(mu: np.ndarray, landmark_position: np.ndarray) -> np.ndarray`
- `measurement_noise_matrix(config: MeasurementModelConfig) -> np.ndarray`
- `innovation_range_bearing(z: np.ndarray, z_hat: np.ndarray) -> np.ndarray`

### Notes

Use a `range_bearing` model first. It is a natural fit for lidar-derived observations and easier to interpret than a more abstract Cartesian residual at this stage.

### Agreed Keywords

- `Measurement Model`
- `Range-Bearing Model`

## Stage 2: Truth Landmark Update Harness

### Goal

Create a controlled update path using known landmark positions from simulation truth before attempting to use extracted landmarks.

### Why This Stage Is Important

This is the safest way to verify that the correction math works. If pose correction fails even when landmark positions are known, then the issue is in the EKF math rather than tracking or association.

### Add In `cave_slam/sim.py`

- config block under `ekf.truth_update`
  - `enabled`
  - `max_observations`
  - `max_range`
- `extract_truth_landmarks_for_update(...)`
- `build_truth_observations(...)`

### Add In `cave_slam/slam.py`

- `simulate_landmark_observations_from_truth(...)`

### Possible Dataclass

- `TruthObservationSet`
  - `observations`
  - `landmark_positions`

### Agreed Keyword

- `Truth Harness`

## Stage 3: Landmark Track Layer

### Goal

Create a proper landmark tracking layer that is distinct from the current visual persistence logic.

### Why This Stage Is Needed

The current TTL-based persistence is adequate for display, but not robust enough to drive estimator updates.

### Add In `cave_slam/slam.py`

- `LandmarkTrack`
  - `track_id`
  - `position`
  - `covariance` optional later
  - `observation_count`
  - `last_seen_frame`
  - `ttl`
  - `quality_score`
- `LandmarkTrackState`
  - `tracks: dict[int, LandmarkTrack]`
  - `next_track_id: int`

### Add Functions

- `initialize_landmark_track_state()`
- `create_landmark_track(point, frame_index, ...)`
- `update_landmark_track(track, point, frame_index, ...)`
- `prune_landmark_tracks(track_state, frame_index, ...)`

### Add In `cave_slam/sim.py`

- extend `SlamState` with a landmark tracking field
- keep `persistent_landmarks` for visualization until the track layer is mature enough to replace it

### Agreed Keyword

- `Track Layer`

## Stage 4: Data Association And Gating

### Goal

Associate current observations to tracked landmarks in a principled way.

### Why This Stage Matters

Correction without reliable association will produce unstable estimates and misleading apparent improvements.

### Add In `cave_slam/slam.py`

- `AssociationConfig`
  - `max_distance`
  - `mahalanobis_threshold`
  - `min_track_quality`
- `AssociationCandidate`
  - `track_id`
  - `innovation`
  - `distance`
  - `mahalanobis_distance`
- `AssociationResult`
  - `matched`
  - `unmatched_observations`
  - `rejected`
- `associate_landmarks_nearest_neighbor(...)`
- `associate_landmarks_mahalanobis(...)`
- `compute_mahalanobis_distance(...)`

### Strategy

Start simple:

1. nearest-neighbor in Euclidean space
2. then add Mahalanobis gating
3. finally reject ambiguous or low-quality matches

### Agreed Keywords

- `Association`
- `Mahalanobis Gating`

## Stage 5: Pose-Only EKF Correction

### Goal

Use associated landmarks to correct the robot pose while leaving landmarks outside the EKF state.

### Why This Stage Is Valuable

It provides a robust intermediate estimator that is far simpler than full EKF-SLAM and easier to validate.

### Add In `cave_slam/slam.py`

- `ekf_update_pose_only(mu, Sigma, observation, landmark_track, measurement_config) -> tuple`
- `ekf_update_pose_only_batch(...)`
- `joseph_covariance_update(...)`

### Possible Dataclass

- `EkfUpdateResult`
  - `mu`
  - `Sigma`
  - `innovation`
  - `kalman_gain`
  - `nis`

### Add In `cave_slam/sim.py`

- a staged pipeline:
  - `predict`
  - `track/update map`
  - `associate`
  - `pose update`

### Agreed Keyword

- `Pose Correction`

## Stage 6: EKF Diagnostics

### Goal

Make the EKF observable and debuggable during development.

### Why This Stage Should Not Be Skipped

As soon as correction is added, tuning becomes difficult without visibility into innovations, matches, and covariance behavior.

### Add In `cave_slam/slam.py`

- `InnovationStats`
  - `innovation_norm`
  - `nis`
  - `accepted`
- `EkfStepDiagnostics`
  - `num_candidate_observations`
  - `num_matches`
  - `num_rejections`
  - `trace_sigma_before`
  - `trace_sigma_after`
  - `pose_update_norm`

### Extend In `cave_slam/sim.py`

- `StepResult`
  - add `ekf_diagnostics`
  - add `association_result`

### Optional Additions In `cave_slam/viz.py`

- text overlays for accepted matches
- innovation or covariance summary
- pose update magnitude display

### Agreed Keywords

- `Diagnostics`
- `NIS`

## Stage 7: Full EKF-SLAM Augmentation

### Goal

Move from pose-only EKF to a joint state containing both robot pose and landmark states.

### Why This Is Deliberately Late

This is the most complex stage. It should only happen after the earlier correction pipeline is stable.

### Add In `cave_slam/slam.py`

- `EkfSlamStateIndex`
  - map from `track_id` to state vector indices
- `augment_state_with_landmark(mu, Sigma, observation, measurement_config) -> tuple`
- `predict_landmark_from_observation(mu_pose, observation) -> np.ndarray`
- `range_bearing_jacobian_full_state(...)`
- `ekf_update_full_state(...)`

### Extend In `cave_slam/sim.py`

- `SlamState`
  - `landmark_indices`
  - `state_track_ids`
- logic for:
  - landmark initialization from first observation
  - covariance augmentation
  - ongoing full-state updates

### Agreed Keywords

- `State Augmentation`
- `Full EKF-SLAM`

## Stage 8: Robustification

### Goal

Improve estimator robustness, tuning stability, and practical usability.

### Typical Enhancements

- outlier rejection
- delayed landmark initialization
- track merge and split logic
- ambiguous association rejection
- configurable update batching
- measurement subsampling
- future loop-closure-like heuristics if desired

### Agreed Keyword

- `Robustification`

## File-By-File Summary

### `cave_slam/slam.py`

Add or extend:

- Stage 0
  - angle normalization and covariance helpers
- Stage 1
  - measurement model and Jacobians
- Stage 3
  - landmark track dataclasses and helpers
- Stage 4
  - association and gating functions
- Stage 5
  - pose-only EKF update functions
- Stage 6
  - diagnostics dataclasses
- Stage 7
  - state augmentation and full-state update logic

### `cave_slam/sim.py`

Add or extend:

- Stage 0
  - EKF debug information in `StepResult`
- Stage 2
  - truth harness configuration and runtime flow
- Stage 3
  - landmark track state in `SlamState`
- Stage 4
  - association orchestration
- Stage 5
  - prediction and correction pipeline
- Stage 6
  - diagnostics collection
- Stage 7
  - augmented state lifecycle management

### `cave_slam/viz.py`

Potential additions:

- Stage 6
  - overlays for matches, covariance size, innovation diagnostics, and EKF mode

### `cave_slam.yaml`

Add a future `ekf:` section with subgroups such as:

- `ekf.mode`
- `ekf.measurement`
- `ekf.association`
- `ekf.truth_update`

### `README.md`

Update after each major stage:

- current EKF mode
- new configuration options
- new debug outputs

## Recommended Execution Order

The safest implementation order is:

1. `Stage 0: EKF Predict Hardening`
2. `Stage 1: Pose-Only Measurement Model`
3. `Stage 2: Truth Landmark Update Harness`
4. `Stage 5: Pose-Only EKF Correction`
5. `Stage 6: EKF Diagnostics`
6. `Stage 3: Landmark Track Layer`
7. `Stage 4: Data Association And Gating`
8. improve `Stage 5: Pose-Only EKF Correction` to use tracked landmarks
9. `Stage 7: Full EKF-SLAM Augmentation`
10. `Stage 8: Robustification`

This order is intentional.

Although `Track Layer` and `Association` are conceptually central, the `Truth Harness` provides a much cleaner way to validate correction math first.

## Suggested “What Stage Are We At?” Phrases

Use any of these phrases when discussing implementation progress:

- `Let's start Stage 0`
- `Implement Predict Hardening`
- `We are at Stage 2 Truth Harness`
- `Move on to Stage 4 Association`
- `Pause Full EKF-SLAM and improve Pose Correction`
- `Add Diagnostics before State Augmentation`

These phrases are intentionally stable so they can be reused accurately in future interactions.

## Summary

The roadmap is built around one core principle:

Do not jump directly from prediction-only EKF to full EKF-SLAM.

Instead, build confidence and capability in this order:

- harden prediction
- add measurement math
- validate correction using truth
- add tracking and association
- perform pose-only correction
- instrument diagnostics
- finally augment the EKF state with landmarks

That is the most logical and robust path for the current `cave_slam` codebase.
