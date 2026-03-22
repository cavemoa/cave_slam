import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cave_slam import DEFAULT_CONFIG_PATH, load_config, run_simulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the cave SLAM simulation from the examples folder."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_simulation(config)


if __name__ == "__main__":
    main()
