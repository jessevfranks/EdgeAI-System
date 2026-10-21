from pathlib import Path

import pytest

from onboard import OnboardConfig


def _write_config(path: Path, fps: str = "3") -> None:
    path.write_text(
        "\n".join(
            (
                "model: model.pt",
                'device: "cpu"',
                "imgsz: 640",
                "confidence: 0.25",
                "iou: 0.7",
                "storage_root: data/onboard",
                f"fps: {fps}",
            )
        ),
        encoding="utf-8",
    )


def test_onboard_config_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "onboard.yaml"
    _write_config(path)

    config = OnboardConfig.from_yaml(path)

    assert config == OnboardConfig(
        model="model.pt",
        device="cpu",
        imgsz=640,
        confidence=0.25,
        iou=0.7,
        storage_root="data/onboard",
        fps=3,
    )


def test_onboard_config_rejects_missing_or_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        OnboardConfig.from_yaml(tmp_path / "missing.yaml")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        OnboardConfig.from_yaml(invalid)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("model: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid onboard configuration"):
        OnboardConfig.from_yaml(malformed)


def test_onboard_config_requires_positive_fps(tmp_path: Path) -> None:
    path = tmp_path / "onboard.yaml"
    _write_config(path, fps="0")

    with pytest.raises(ValueError, match="greater than zero"):
        OnboardConfig.from_yaml(path)
