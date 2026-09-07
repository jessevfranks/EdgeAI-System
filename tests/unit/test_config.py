from pathlib import Path

import pytest

from edge_ai.config import (
    CORE_WIDE_SEARCH_SPACE,
    ConfigurationError,
    ExperimentConfig,
    ModelScale,
    dataset_fingerprint,
    model_checkpoint,
)


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("train: images/train\nval: images/val\ntest: images/test\nnames: [target]\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("scale", ["n", "s", "m", "l", "x"])
def test_model_scale_checkpoint_mapping(scale: str) -> None:
    assert model_checkpoint(scale) == f"yolov8{scale}.pt"


def test_invalid_scale_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ModelScale.parse("xxl")


def test_tuning_profiles_resolve_and_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(action="tune", data=str(_dataset(tmp_path)), scale=ModelScale.NANO, name="test")
    assert config.search_bounds() == CORE_WIDE_SEARCH_SPACE
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored.scale is ModelScale.NANO
    assert restored.iterations == 10


def test_custom_bounds_must_be_ordered(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        ExperimentConfig(
            action="tune",
            data=str(_dataset(tmp_path)),
            scale=ModelScale.NANO,
            name="bad",
            search_profile="custom",
            search_space={"lr0": (0.1, 0.01)},
        )


def test_dataset_fingerprint_changes_with_descriptor(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    first = dataset_fingerprint(dataset)
    dataset.write_text("train: a\nval: b\ntest: c\nnames: [target, other]\n", encoding="utf-8")
    assert dataset_fingerprint(dataset) != first


def test_training_config_serializes_paths(tmp_path: Path) -> None:
    config = ExperimentConfig(
        action="train", data=str(_dataset(tmp_path)), scale=ModelScale.SMALL, name="baseline"
    )
    payload = config.to_dict()
    assert payload["scale"] == "s"
    assert isinstance(payload["database_path"], str)
