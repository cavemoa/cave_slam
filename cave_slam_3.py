import argparse

from cave_slam import DEFAULT_CONFIG_PATH, load_config, run_simulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the refactored cave SLAM simulation."
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
