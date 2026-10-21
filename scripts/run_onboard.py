"""Run the onboard pipeline against finite test inputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    test_images = subcommands.add_parser(
        "test-images",
        help="process each supported image in a directory once",
    )
    test_images.add_argument("directory", type=Path)
    test_images.add_argument("--config", type=Path, default=Path("configs/onboard.yaml"))
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "src"))
    os.chdir(project)

    from onboard import OnboardConfig, OnboardOrchestrator, PacedImageDirectory

    args = parse_args(arguments)
    try:
        config = OnboardConfig.from_yaml(args.config)
        source = PacedImageDirectory(args.directory, config.fps)
        orchestrator = OnboardOrchestrator(
            model_path=config.model,
            storage_root=config.storage_root,
            device=config.device,
            imgsz=config.imgsz,
            confidence=config.confidence,
            iou=config.iou,
        )
        for result in orchestrator.run_stream(source):
            frame_id = result.acquired_frame.metadata.frame_id
            detections = len(result.inference_result.detections)
            print(f"Processed frame {frame_id}: {detections} detections")
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
