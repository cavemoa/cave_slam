# EKF In Cave SLAM

This document describes how the Extended Kalman Filter is currently implemented in the `cave_slam` simulation.

It is intentionally practical rather than abstract. The goal is to explain:

- what state the estimator keeps
- how prediction and correction are performed
- how landmark tracks and data association interact with the filter
- how `pose_only` and `full_slam` differ
- what diagnostics are available
- what limitations still exist

## Scope

The estimator in this project is a lightweight educational EKF implementation for a 2D simulation. It is not intended to be a production robotics stack.

The current implementation lives mainly in:

- `cave_slam/slam.py`
- `cave_slam/ekf.py`
- `cave_slam/sim.py`
- `cave_slam/viz.py`

## Two EKF Modes

The estimator supports two runtime modes through `ekf.mode` in the config:

- `pose_only`
- `full_slam`

### `pose_only`

In `pose_only` mode, the EKF state contains only the robot pose:

```text
mu = [x, y, theta]
```

This mode supports two correction sources:

- truth-harness observations
- associated landmark tracks

Truth-harness observations are useful for controlled validation because the landmark positions are known exactly from the simulation world.

### `full_slam`

In `full_slam` mode, the EKF state is augmented with landmark positions:

```text
mu = [x, y, theta, l1x, l1y, l2x, l2y, ...]
```

Each landmark in the EKF state is tied to a landmark-track ID from the track layer.

In this mode:

- matched landmarks already in the state receive full-state EKF updates
- newly created landmark tracks can be inserted into the EKF state by augmentation

## Runtime State

The runtime estimator state is split across several objects.

### `SlamState`

`cave_slam/sim.py` stores the live estimator data in `SlamState`.

Important fields are:

- `mu`
  - EKF mean vector
- `Sigma`
  - EKF covariance matrix
- `landmark_track_state`
  - external landmark tracks used for association
- `ekf_slam_index`
  - mapping between landmark-track IDs and positions inside the augmented EKF state

### `LandmarkTrackState`

The landmark track layer is separate from the EKF state. It exists to provide a stable association target and to support later feature development.

Each `LandmarkTrack` stores:

- `track_id`
- `position`
- `covariance`
- `observation_count`
- `last_seen_frame`
- `ttl`
- `quality_score`

### `EkfSlamStateIndex`

This is the bridge between the track layer and the augmented EKF state.

It stores:

- `track_id_to_index`
  - map from landmark-track ID to the start index of that landmark inside `mu`
- `state_track_ids`
  - ordered list of track IDs that have been augmented into the EKF state

If a track has not yet been augmented, it may still exist in the track layer, but it will not yet have an entry in the EKF state index.

## Prediction Step

The prediction step is implemented in `cave_slam/slam.py` by `ekf_predict(...)`.

It uses noisy odometry from the agent motion command:

- `measured_distance`
- `measured_turn`

The state update is:

```text
theta' = theta + dtheta
x' = x + d * cos(theta')
y' = y + d * sin(theta')
```

Only the pose part of the state is directly advanced. If the EKF state is augmented, landmark positions remain fixed during prediction.

The covariance update uses:

- motion Jacobian `G`
- control Jacobian `V`
- control noise covariance `Q`

The implementation includes numerical hardening:

- angle normalization
- covariance symmetrization
- positive-semidefinite repair

## Measurement Model

The measurement model is a range-bearing observation of a landmark:

```text
z = [range, bearing]
```

This is implemented in `cave_slam/slam.py` using:

- `predict_range_bearing(...)`
- `range_bearing_jacobian_pose(...)`
- `range_bearing_jacobian_landmark(...)`
- `range_bearing_jacobian_full_state(...)`
- `measurement_noise_matrix(...)`
- `innovation_range_bearing(...)`

### Predicted Measurement

For robot pose `(x, y, theta)` and landmark `(lx, ly)`:

```text
dx = lx - x
dy = ly - y
range = sqrt(dx^2 + dy^2)
bearing = atan2(dy, dx) - theta
```

The bearing innovation is always angle-wrapped into `[-pi, pi)`.

## Pose-Only Correction

Pose-only correction is implemented with:

- `ekf_update_pose_only(...)`
- `ekf_update_pose_only_batch(...)`

These functions keep the EKF state size fixed at `3`.

They use:

- the current observation
- a known or associated landmark world position
- the standard EKF update equations
- Joseph-form covariance update for better numerical behavior

This mode is useful when:

- validating the measurement-update math
- comparing truth-harness correction to tracked-landmark correction
- keeping the estimator simpler than full EKF-SLAM

## Landmark Track Layer

The landmark track layer is implemented in `cave_slam/slam.py` and orchestrated from `cave_slam/ekf.py`.

Its purpose is to provide:

- stable landmark identities across frames
- a clean separation between display-only landmarks and estimator landmarks
- a base for association and later EKF-SLAM augmentation

The track layer is updated from extracted scan features.

The landmark model is now explicit rather than implicit. Observations and tracks both carry a `landmark_type`, and the active pipeline can now emit `corner`, `line_segment`, `endpoint`, and `junction` landmarks.

For estimator stability, there is still a boundary:

- `corner`, `endpoint`, and `junction` are treated as point-compatible EKF features
- `line_segment` is currently kept external to the EKF state and used only in the typed track layer

`Stage H: Type-Aware EKF` builds on that boundary rather than removing it. The current estimator still uses a point landmark model for `corner`, `endpoint`, and `junction`, but it no longer treats those three types identically:

- `junction` observations are prioritized first for EKF correction and augmentation
- `endpoint` observations are prioritized next
- `corner` observations remain valid but are treated as the least-structured point features
- measurement noise is scaled by landmark type, so endpoints and junctions can be given tighter effective noise than corners

The typed track layer is now also type-aware in its matching rules:

- `corner`, `endpoint`, and `junction` use point-style spatial association
- `line_segment` uses midpoint position plus orientation and extent gating

It currently supports:

- track creation
- nearest-track position update
- TTL-style persistence
- quality scoring
- pruning of non-augmented stale tracks
- protection of augmented tracks from pruning

The last point matters in `full_slam` mode. Once a track has been inserted into the EKF state, the track layer keeps that track alive so the EKF state index does not silently lose its anchor.

## Data Association

Association is implemented in `cave_slam/slam.py` and called from `associate_feature_observations(...)` in `cave_slam/ekf.py`.

Two association modes are supported:

- `nearest_neighbor`
- `mahalanobis`

Both modes now also support ambiguity rejection.

### Nearest-Neighbor Association

This mode uses a type-aware distance score:

- `corner`, `endpoint`, and `junction`: Euclidean distance between observed and tracked world position
- `line_segment`: midpoint distance plus orientation and extent penalties, with hard rejection if the mismatch exceeds configured thresholds

If the track has already been augmented into the EKF state, the state landmark position is used instead of the plain track centroid.

### Mahalanobis Association

This mode uses:

- the predicted range-bearing residual
- the innovation covariance
- a Mahalanobis distance threshold

If a landmark is already augmented into the EKF state, the full-state measurement Jacobian is used. If not, association falls back to a pose-only landmark prediction against the external track position.

This means Mahalanobis association naturally gets more informative once landmarks have entered the EKF state.

For `line_segment` tracks, Mahalanobis association still uses midpoint range-bearing residuals only after the line has passed the type-aware orientation and extent gates. Line segments remain external to the EKF state in the current implementation.

Because Mahalanobis association uses the EKF measurement noise matrix, the new type-aware measurement scaling also affects the association score:

- `junction` tracks can produce tighter innovations than corners
- `endpoint` tracks can also be weighted slightly more strongly than corners
- `line_segment` tracks still remain external and are never augmented into the EKF state

### Ambiguity Rejection

After finding the best association candidate, the estimator can reject the match if the second-best candidate is too similar.

This is intended to catch cases where:

- two nearby tracks both look plausible
- the best candidate is only marginally better than the runner-up
- forcing a match would be more dangerous than skipping the update

The current implementation supports two ambiguity tests:

- ratio test
  - reject if `second_best / best` is too small
- margin test
  - reject if `second_best - best` is too small

For nearest-neighbor association, the score is Euclidean distance.

For Mahalanobis association, the score is Mahalanobis distance.

If either test indicates that the top two candidates are too close, the observation is treated as unmatched and counted as an ambiguous rejection rather than a normal matched update.

Ambiguous observations are not fed into the landmark-track update path either. That prevents a weakly distinguished feature from quietly creating or refreshing an external track which could later be augmented into the EKF state.

### Association Output

Association returns an `AssociationResult` containing:

- `matched`
- `unmatched_observations`
- `rejected`
- `method`
- `gating_applied`

The simulation uses this result in two places:

- to update the track layer consistently
- to drive pose-only or full-state EKF correction

## Track Update Result

After association, the track layer is updated using `update_landmark_track_state(...)`.

This returns a `TrackUpdateResult` with:

- updated `track_state`
- `assignments`

Each assignment records:

- `observation_index`
- `track_id`
- `created`

This is important in `full_slam` mode because newly created tracks can be promoted directly into the EKF state.

## Full EKF-SLAM Augmentation

Full-state augmentation is implemented in `augment_state_with_landmark(...)` in `cave_slam/slam.py`.

When a landmark is first inserted into the EKF state:

1. The landmark world position is reconstructed from the current robot pose and the observation.
2. The state vector is extended by two elements.
3. The covariance matrix is enlarged.
4. Cross-covariances between robot pose and landmark are initialized from the pose covariance and measurement noise.

The augmentation path is now type-aware in two ways:

1. Candidate tracks are considered in landmark-type priority order:
   - `junction`
   - `endpoint`
   - `corner`
2. The landmark measurement noise used in covariance initialization is scaled by landmark type.

That means well-formed endpoint and junction tracks can both enter the EKF earlier and be initialized with slightly tighter uncertainty than generic corners, if the config is tuned that way.

### Landmark Initialization

For observation `(r, b)`:

```text
phi = theta + b
lx = x + r cos(phi)
ly = y + r sin(phi)
```

### Covariance Augmentation

The implementation uses pose and measurement Jacobians to build:

- pose-to-landmark covariance
- landmark self-covariance
- full cross-covariance with the existing EKF state

This ensures the new landmark is correlated with the robot pose and any previously correlated landmarks.

## Full-State Landmark Updates

Once a landmark has been augmented, future matched observations can update the full EKF state using `ekf_update_full_state(...)`.

In that case:

- the measurement Jacobian includes both pose and landmark terms
- the full covariance matrix participates in the update
- the correction can affect both robot pose and landmark coordinates

This is the key difference between `pose_only` and `full_slam`.

In `pose_only`, landmarks are external references.

In `full_slam`, landmarks become part of the joint estimated state.

## How `step_simulation()` Uses The EKF

The estimator orchestration now happens mainly in `cave_slam/ekf.py`, while `cave_slam/sim.py` coordinates the larger simulation frame.

At a high level, each simulation step currently does the following:

1. Copy the current true observation pose.
2. Optionally build truth-harness observations.
3. Simulate lidar.
4. Extract typed landmarks from the scan.
5. Build feature observations in range-bearing and world-point form.
6. Associate feature observations against the track layer.
7. Update the visual landmark layer.
8. Update the landmark track layer using the association result.
9. Run either:
    - pose-only correction, or
    - full-state augmentation and update
10. Sync augmented landmark positions back into the track layer in `full_slam` mode.
11. Update the accumulated point cloud and voxel map using the corrected observation-time estimate.
12. Step the agent and generate noisy odometry.
13. Run EKF prediction to advance the state to the post-motion estimate.
14. Compute EKF diagnostics.

This is now an explicit `observe -> update -> map -> move -> predict` timing model.

That is more internally consistent for this simulation than the earlier pattern where observations came from the pre-motion pose but correction happened only after prediction.

When multiple matched observations are available in a frame, the correction stage now processes them in confidence order:

- Mahalanobis association: lowest Mahalanobis distance first
- nearest-neighbor association: lowest Euclidean distance first

That makes the limited `max_updates_per_frame` budget more useful and reduces the influence of weaker matches.

## How Full-SLAM Correction Works In Practice

In `full_slam` mode, the current implementation behaves as follows:

- if an associated track is already in the EKF state:
  - apply a full-state EKF update
- if a track is not yet in the EKF state:
  - keep it external until it is stable enough for augmentation
- once an external track satisfies the augmentation stability rule:
  - augment the state with that landmark

This means the EKF state gradually grows as the agent accumulates distinct landmark tracks, but no longer grows immediately on first sighting.

### Delayed Landmark Initialization

Delayed landmark initialization is now part of `full_slam`.

The goal is to avoid polluting the augmented EKF state with weak one-off detections.

A landmark track must stay external until it satisfies at least one of these conditions:

- `track.observation_count >= ekf.augmentation.min_observations`
- `track.quality_score >= ekf.augmentation.min_track_quality`

Only then is it promoted from the external track layer into the augmented EKF state.

This makes the full-state estimator more conservative:

- new detections are tracked first
- unstable or fleeting landmarks remain outside the EKF state
- only more reliable tracks become permanent EKF landmarks

## Diagnostics

Diagnostics are implemented in `cave_slam/slam.py` and carried in `StepResult` by `cave_slam/sim.py`.

The key diagnostics types are:

- `EkfDebugInfo`
- `InnovationStats`
- `EkfStepDiagnostics`

### `EkfDebugInfo`

Contains compact pose covariance summaries:

- `trace_sigma`
- `pose_std_x`
- `pose_std_y`
- `pose_std_theta`

### `EkfStepDiagnostics`

Contains:

- `num_candidate_observations`
- `num_matches`
- `num_rejections`
- `ambiguous_rejections`
- `trace_sigma_before`
- `trace_sigma_after`
- `pose_update_norm`
- `innovation_stats`
- `mean_nis`
- `max_nis`

### Live Overlay

The Matplotlib visualization shows a compact EKF panel with:

- current EKF mode
- state size
- track count
- number of landmark augmentations in the current frame
- association method and counts
- ambiguous rejection count
- covariance trace change
- pose update magnitude
- NIS summary
- pose standard deviations

## Config Fields

The EKF-related config currently includes:

### `ekf.mode`

- `pose_only`
- `full_slam`

### `ekf.measurement`

- `model_type`
- `range_std`
- `bearing_std_deg`
- `corner_noise_scale`
- `endpoint_noise_scale`
- `junction_noise_scale`

The scale fields are multiplicative factors applied to both the base range and base bearing standard deviations when the observation type is `corner`, `endpoint`, or `junction`.

This type-aware measurement noise is used in:

- Mahalanobis association
- pose-only correction
- full-state correction
- state augmentation covariance initialization

### `ekf.truth_update`

- `enabled`
- `max_observations`
- `max_range`

### `ekf.pose_update`

- `enabled`
- `use_truth_observations`
- `max_updates_per_frame`
- `nis_threshold`

`nis_threshold` controls NIS-based outlier rejection at the EKF correction stage.

In practical terms:

- if a candidate update produces `NIS <= nis_threshold`
  - the update is accepted
- if a candidate update produces `NIS > nis_threshold`
  - the update is rejected and the EKF state is left unchanged for that observation

This gate is applied:

- in `pose_only` correction
- in full-state landmark updates for `full_slam`

It is not the same thing as association gating:

- association gating decides whether an observation matches a track
- NIS gating decides whether the matched observation should actually be allowed to update the filter

The default threshold in this project is intentionally more permissive than a strict textbook chi-square threshold, because the simulation uses an explicit `observe -> update -> move -> predict` loop rather than a stricter robotics middleware timing model.

In `full_slam` mode, the correction path uses associated track observations rather than the truth harness.

### `ekf.association`

- `method`
- `max_distance`
- `mahalanobis_threshold`
- `min_track_quality`
- `ambiguity_ratio_threshold`
- `ambiguity_margin_threshold`

`ambiguity_ratio_threshold` and `ambiguity_margin_threshold` are used to reject associations whose best and second-best candidates are too similar.

This is separate from:

- Mahalanobis gating
- NIS-based EKF update rejection

### `ekf.augmentation`

- `min_observations`
- `min_track_quality`
- `endpoint_min_observations`
- `endpoint_min_track_quality`
- `junction_min_observations`
- `junction_min_track_quality`

These settings control delayed landmark initialization for `full_slam`.

They decide when an external landmark track is considered stable enough to be augmented into the EKF state.

The readiness rule is now type-aware:

- `corner` tracks use the base thresholds
- `endpoint` tracks use the endpoint-specific thresholds
- `junction` tracks use the junction-specific thresholds

This makes it possible to let sharper structural features enter the state earlier without relaxing augmentation for all landmark types.

## Practical Tuning Notes

If `full_slam` appears to augment landmarks but not update them often:

- increase `ekf.association.max_distance`
- increase `ekf.association.ambiguity_ratio_threshold` or `ekf.association.ambiguity_margin_threshold` if ambiguity rejection is too aggressive
- try `nearest_neighbor` first before `mahalanobis`
- lower `ekf.association.min_track_quality`
- increase sensor field of view or number of rays
- simplify the environment so the same features are seen repeatedly
- lower `ekf.augmentation.min_observations` or `ekf.augmentation.min_track_quality` if augmentation is too conservative

If the estimator becomes unstable:

- lower `max_updates_per_frame`
- reduce measurement noise optimism by increasing `range_std` or `bearing_std_deg`
- use `pose_only` first to validate the scan features and association settings

If the map grows too quickly in `full_slam` mode:

- tighten association thresholds
- reduce spurious landmark extraction
- increase feature quality by tuning the feature-extraction settings

## Current Limitations

The EKF implementation is now substantially more capable than the original prediction-only version, but several limitations remain.

- Landmark extraction is still heuristic and currently derives all feature types directly from local scan geometry.
- The track layer does not yet support merge or split logic.
- Augmented landmarks are not currently removed from the EKF state.
- There is no loop-closure system or smoothing back-end.
- The simulation loop is still designed for experimentation and visualization rather than a decoupled robotics middleware architecture.
- Association quality remains sensitive to feature-extraction and environment tuning.

## What To Read Next

For the staged development history and remaining roadmap:

- `documentation/ekf_roadmap.md`

For a broader overview of the whole project:

- `README.md`
