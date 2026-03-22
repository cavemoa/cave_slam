# Cave SLAM

`Cave SLAM` is a small 2D simulation project for exploring SLAM-related ideas in a visually inspectable way. It combines:

- procedural cave-like environment generation
- a simple 2D lidar model with configurable noise
- corner-style feature extraction from lidar returns
- a basic EKF prediction step driven by noisy odometry
- accumulated point-cloud mapping with voxel-grid averaging
- a reactive agent controller with obstacle avoidance

The project is intentionally lightweight. It is designed to be easy to modify, inspect, and use for experimentation rather than to serve as a production robotics stack.

## Project Status

The current refactored entry point is:

- [cave_slam_3.py](/home/jon/cave_slam/cave_slam_3.py)

The original monolithic reference version is preserved here:

- [cave_slam_2.py](/home/jon/cave_slam/cave_slam_2.py)

Older exploratory scripts are stored in:

- [old_scripts](/home/jon/cave_slam/old_scripts)

## Repository Layout

```text
cave_slam/
├── README.md
├── cave_slam.yaml
├── cave_slam_2.py
├── cave_slam_3.py
├── examples/
│   ├── headless_experiment.py
│   └── visual_runner.py
├── requirements.txt
├── cave_slam/
│   ├── __init__.py
│   ├── agent.py
│   ├── sim.py
│   ├── slam.py
│   └── viz.py
└── old_scripts/
```

### Module Responsibilities

- [cave_slam/slam.py](/home/jon/cave_slam/cave_slam/slam.py)
  - lidar ray casting and measurement generation
  - sensor noise model
  - landmark extraction from local scan geometry
  - point-cloud transformation
  - voxel-grid aggregation
  - EKF prediction math

- [cave_slam/agent.py](/home/jon/cave_slam/cave_slam/agent.py)
  - agent motion state
  - path-clear / bumper logic
  - weighted freer-side obstacle avoidance
  - startup spin behavior
  - movement stepping

- [cave_slam/sim.py](/home/jon/cave_slam/cave_slam/sim.py)
  - typed configuration models and validation
  - procedural environment generation
  - runtime state containers
  - headless simulation stepping
  - simulation orchestration without plotting concerns

- [cave_slam/viz.py](/home/jon/cave_slam/cave_slam/viz.py)
  - Matplotlib backend setup
  - plotting artist creation
  - rendering of simulation state
  - animation loop and interactive runtime orchestration

- [cave_slam_3.py](/home/jon/cave_slam/cave_slam_3.py)
  - minimal CLI runner that loads a config and starts the simulation

- [examples/headless_experiment.py](/home/jon/cave_slam/examples/headless_experiment.py)
  - demonstrates headless stepping and prints a textual run summary

- [examples/visual_runner.py](/home/jon/cave_slam/examples/visual_runner.py)
  - demonstrates running the interactive visualization from the examples folder

## Requirements

The project currently depends on:

- `matplotlib`
- `numpy`
- `PyYAML`

These are listed in [requirements.txt](/home/jon/cave_slam/requirements.txt).

## Architecture

The current refactor separates the project into three layers:

- domain logic
  - [cave_slam/slam.py](/home/jon/cave_slam/cave_slam/slam.py) and [cave_slam/agent.py](/home/jon/cave_slam/cave_slam/agent.py)
- headless simulation engine
  - [cave_slam/sim.py](/home/jon/cave_slam/cave_slam/sim.py)
- visualization and interactive runtime
  - [cave_slam/viz.py](/home/jon/cave_slam/cave_slam/viz.py)

This means you can:

- run the interactive Matplotlib simulation through [cave_slam_3.py](/home/jon/cave_slam/cave_slam_3.py)
- import the package without triggering Matplotlib setup
- create a simulation and advance it frame-by-frame in headless code using `create_simulation()` and `step_simulation()`

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the simulation

```bash
python cave_slam_3.py
```

To use a specific config file:

```bash
python cave_slam_3.py --config cave_slam.yaml
```

If you prefer to call the virtualenv interpreter directly:

```bash
.venv/bin/python cave_slam_3.py --config cave_slam.yaml
```

### 4. Try the examples

Headless experiment:

```bash
python examples/headless_experiment.py --config cave_slam.yaml --steps 200 --summary-every 50
```

Visual runner:

```bash
python examples/visual_runner.py --config cave_slam.yaml
```

## How the Simulation Works

At a high level, each animation frame follows this loop:

1. The agent pose is used to simulate a lidar scan against the wall geometry.
2. The scan is converted into local measurements.
3. Corner-like landmarks are extracted from the scan.
4. Measurements are transformed into world coordinates using the current EKF pose.
5. The point cloud is accumulated and averaged into a voxel grid.
6. The agent controller chooses a motion command.
7. Odometry noise is added to the command.
8. The EKF prediction step updates the estimated pose.
9. Matplotlib artists are refreshed for display.

The first eight steps are now handled by the headless simulation engine in [cave_slam/sim.py](/home/jon/cave_slam/cave_slam/sim.py). The final display step is handled separately in [cave_slam/viz.py](/home/jon/cave_slam/cave_slam/viz.py).

### Environment Model

The environment is a bounded 2D world made of line-segment walls.

Two modes are supported:

- generated environment
  - random polygons are placed inside the bounding box
- manual wall list
  - explicit wall segments are supplied in YAML

The generator can be controlled through complexity, world size, polygon area range, and boundary overlap behavior.

### Lidar Model

The lidar model in [cave_slam/slam.py](/home/jon/cave_slam/cave_slam/slam.py) casts rays across a configurable field of view. For each ray:

- the nearest wall intersection is found
- optional range-dependent noise is added
- hit measurements are retained for mapping and feature extraction
- full scan samples are retained for obstacle-avoidance logic

The forward sector is also tracked so the controller can estimate how close an obstacle is directly ahead.

### Landmark Extraction

The current feature extraction stage looks for corner-like structures in the ordered lidar returns. It does this by:

- converting neighboring scan samples into local 2D points
- fitting line directions to a small left and right window
- checking span, continuity, residual error, and corner angle
- applying non-maximum suppression to avoid dense duplicate detections

Detected landmarks are persisted across frames with a simple TTL-based association rule.

### Mapping

The simulation keeps two related map views:

- raw accumulated point cloud
- voxel-grid averaged points

The voxel grid reduces visual clutter by averaging repeated returns that fall into the same spatial cell once a minimum point count is reached.

### Motion and Obstacle Avoidance

The agent controller in [cave_slam/agent.py](/home/jon/cave_slam/cave_slam/agent.py) supports:

- startup spin-in-place behavior
- random wandering when the path is clear
- obstacle-triggered avoidance with hysteresis
- a simple bumper check before committing motion

The current avoidance strategy is a weighted freer-side selector:

- left and right scan sectors are scored separately
- sector scores are weighted by `cos(angle)^2`, so rays closer to straight ahead matter more
- the agent chooses the side with greater available clearance
- once chosen, that direction is retained until the obstacle is considered cleared

This avoids the per-frame oscillation that occurs if the controller randomly re-chooses left or right every update.

### EKF State Prediction

The EKF implementation in this project is prediction-only. It uses:

- commanded distance with added noise
- commanded turn with added noise

to propagate:

- estimated pose `mu`
- pose covariance `Sigma`

There is currently no explicit measurement-update stage for landmark corrections. That means the EKF estimate is essentially an odometry-driven prediction track rather than a full closed-loop SLAM estimator.

## Running the Refactored Version

The recommended script is [cave_slam_3.py](/home/jon/cave_slam/cave_slam_3.py):

```bash
python cave_slam_3.py
```

It simply:

1. parses `--config`
2. loads and validates the YAML config against typed config models
3. calls `run_simulation(config)` from [cave_slam/viz.py](/home/jon/cave_slam/cave_slam/viz.py)

If you want to drive the simulation programmatically, the public package entry points are exposed from [cave_slam/__init__.py](/home/jon/cave_slam/cave_slam/__init__.py):

```python
from cave_slam import load_config, run_simulation

config = load_config("cave_slam.yaml")
run_simulation(config)
```

For headless stepping without plotting:

```python
from cave_slam import create_simulation, load_config, step_simulation

config = load_config("cave_slam.yaml")
state = create_simulation(config)

for _ in range(100):
    result = step_simulation(state)
```

`step_simulation()` returns a typed `StepResult` object containing the frame observation pose, lidar scan, extracted landmarks, chosen motion command, and noisy odometry values.

## Configuration

Runtime configuration is loaded from YAML, merged with defaults, and converted into typed configuration objects defined in [cave_slam/sim.py](/home/jon/cave_slam/cave_slam/sim.py). The default file in the repository is [cave_slam.yaml](/home/jon/cave_slam/cave_slam.yaml).

The main typed config root is `AppConfig`, which contains nested config sections such as:

- `SimulationConfig`
- `EnvironmentConfig`
- `SensorConfig`
- `FeatureExtractionConfig`
- `MotionConfig`
- `OdometryNoiseConfig`
- `VoxelGridConfig`
- `AgentConfig`
- `PlotConfig`

### `simulation`

- `delay_ms`
  - delay between animation frames in milliseconds
- `frames`
  - total number of animation frames
- `random_seed`
  - random seed for reproducible runs

### `environment.generator`

- `enabled`
  - when `true`, use procedural environment generation
- `complexity`
  - approximate number of polygonal obstacles to create
- `width`
  - world width
- `height`
  - world height
- `overlap_boundaries`
  - whether generated obstacles may extend across the room boundary
- `min_area`
  - minimum polygon area
- `max_area`
  - maximum polygon area

### `environment.walls`

Used when `environment.generator.enabled: false`.

Each wall is a line segment of the form:

```yaml
walls:
  - [[0, 0], [10, 0]]
  - [[10, 0], [10, 10]]
```

### `sensor`

- `fov_degrees`
  - lidar field of view in degrees
- `num_rays`
  - number of rays per scan
- `max_range`
  - maximum measurable distance

### `sensor.noise`

- `enabled`
  - enables range noise
- `min_relative_std`
  - lower bound on relative range standard deviation
- `max_relative_std`
  - upper bound on relative range standard deviation

The sensor noise grows with distance.

### `feature_extraction`

- `window_size`
  - number of neighboring points used on each side of a candidate feature
- `max_neighbor_gap`
  - maximum allowed gap between neighboring scan points
- `min_segment_span`
  - minimum spatial extent of each side of a candidate corner
- `max_line_residual`
  - maximum acceptable point-to-line deviation
- `min_corner_angle_deg`
  - minimum allowed corner angle
- `max_corner_angle_deg`
  - maximum allowed corner angle
- `nms_radius`
  - non-maximum suppression radius in scan index space
- `persistence_frames`
  - lifetime of landmarks once detected
- `association_radius`
  - spatial threshold for associating landmarks across frames

### `motion`

- `linear_speed`
  - forward distance per frame during normal motion
- `random_turn_deg`
  - random steering jitter applied during free motion
- `collision_turn_deg`
  - turn applied during obstacle avoidance
- `collision_distance`
  - threshold for detecting an obstacle ahead
- `clear_distance`
  - distance required before avoidance can be exited
- `clearance_frames`
  - number of consecutive clear frames required to exit avoidance
- `front_sector_deg`
  - forward angular band used for obstacle detection and side scoring separation
- `side_balance_tolerance`
  - threshold below which left and right sector scores are treated as effectively tied

Note:
The current repository YAML does not include every motion field. Missing values are filled from code defaults in [cave_slam/sim.py](/home/jon/cave_slam/cave_slam/sim.py).

### `odometry_noise`

- `distance_std`
  - standard deviation of commanded distance noise
- `angle_std_deg`
  - standard deviation of commanded turn noise in degrees

### `voxel_grid`

- `voxel_size`
  - spatial grid cell size
- `min_points_per_voxel`
  - minimum number of points before a voxel contributes to the displayed map
- `point_size`
  - marker size of voxelized map points
- `point_alpha`
  - transparency of voxelized points
- `color`
  - Matplotlib color name for voxel points

### `agent.initial_pose`

- `x`
  - initial x position
- `y`
  - initial y position
- `theta_deg`
  - initial heading in degrees

### `agent.startup_behavior`

- `enabled`
  - enables startup behavior
- `mode`
  - currently `spin_in_place`
- `rotation_degrees`
  - total startup rotation to perform
- `turn_step_deg`
  - incremental turn per frame during startup

### `plot`

- `figsize`
  - Matplotlib figure size
- `xlim`
  - x-axis display limits
- `ylim`
  - y-axis display limits
- `point_size`
  - raw point cloud marker size
- `point_alpha`
  - raw point cloud transparency

## Outputs and Visual Elements

The visualization currently shows:

- black line segments for walls
- blue points for the raw accumulated point cloud
- orange points for the voxel-grid map
- red outlined circles for detected persistent landmarks
- a dashed green line for the true trajectory
- a magenta line for the EKF predicted trajectory
- a red marker and heading for the true pose
- a magenta marker and heading for the estimated pose

## Design Notes

This project intentionally keeps the math and simulation logic exposed in plain Python. That makes it useful for:

- teaching
- algorithm comparison
- quick prototyping
- experimenting with controller and sensor settings

The refactor into `slam.py`, `agent.py`, and `sim.py` was done to reduce complexity in the main runner and make future iteration safer.

More recently, plotting was separated again into [cave_slam/viz.py](/home/jon/cave_slam/cave_slam/viz.py), and configuration/state were converted into typed dataclasses. The result is a package that is easier to:

- test without a GUI
- run headlessly in scripts
- inspect in an IDE
- extend with new controllers, estimators, or rendering layers

## Known Limitations

- The EKF currently performs prediction only and does not yet incorporate a correction step from landmarks.
- The map is an accumulated visualization, not a globally optimized SLAM backend.
- Obstacle avoidance is reactive rather than goal-directed.
- The simulation is 2D only.
- The interactive runtime is still tied to Matplotlib animation, so the visual mode is best suited to local use.

## Suggested Next Steps

If you want to extend the project, the most natural next improvements are:

- add an EKF measurement-update stage using persistent landmarks
- add data association diagnostics and visualization
- expose more controller parameters in the YAML file
- save trajectories and map snapshots for offline analysis
- add unit tests for geometry, lidar, and controller helpers
- add a headless batch mode for repeated experiments

## Troubleshooting

### `ModuleNotFoundError` for plotting or YAML packages

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Matplotlib cache or config warnings

If Matplotlib complains about a non-writable config directory, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

before running the simulation.

### No GUI window appears

Possible causes:

- running in a headless environment
- missing GUI backend support
- remote session without display forwarding

The code attempts to choose a sensible Matplotlib backend, but interactive plotting still depends on the local environment.

## License

No license file is currently included in the repository. If you intend to distribute or share the project, add an explicit license.
