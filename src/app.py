"""Main entry point for the AR Assembly Copilot."""

import argparse
import yaml

from src.pipeline.orchestrator import Pipeline, PipelineConfig


def load_config(config_path: str) -> PipelineConfig:
    """Load pipeline config from a YAML file."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    config = PipelineConfig()
    if data:
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


def main():
    parser = argparse.ArgumentParser(description="AR Assembly Copilot")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--camera", "-cam",
        type=int,
        default=None,
        help="Camera index (overrides config)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to YOLO model weights (overrides config)",
    )
    parser.add_argument(
        "--fullscreen", "-f",
        action="store_true",
        help="Run in fullscreen mode",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable logging",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig()

    # Apply CLI overrides
    if args.camera is not None:
        config.camera_source = args.camera
    if args.model is not None:
        config.model_path = args.model
    if args.fullscreen:
        config.fullscreen = True
    if args.no_log:
        config.enable_logging = False

    # Run
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
