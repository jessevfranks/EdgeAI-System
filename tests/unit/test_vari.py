import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pytest
from PIL import Image

from onboard import Detection, FrameMetadata, InferenceResult, VariCalculator


def make_result(tmp_path, pixels, boxes):
    path = tmp_path / "processed.png"
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(path)
    return InferenceResult(
        metadata=FrameMetadata(7, datetime(2026, 10, 13, tzinfo=UTC)),
        model_path=tmp_path / "model.engine",
        processed_path=path,
        annotated_path=tmp_path / "annotated.png",
        json_path=tmp_path / "inference.json",
        detections=tuple(
            Detection(*box, index, f"plant_{index}", 0.9)
            for index, box in enumerate(boxes)
        ),
    )


def test_values_storage(tmp_path):
    source = make_result(
        tmp_path,
        [[[10, 30, 20], [30, 10, 20], [10, 30, 35], [30, 10, 45], [30, 10, 35],
          [10, 20, 30], [0, 0, 0]]],
        [(0, 0, 7, 1), (1, 0, 3, 1)],
    )
    result = VariCalculator(tmp_path).compute(source)
    first, second = result.detections
    expected = np.array([[1, -1, 4, 4, -4, np.nan, np.nan]], dtype=np.float32)
    np.testing.assert_allclose(first.pixels, expected)
    np.testing.assert_allclose(second.pixels, [[-1, 4]])
    assert first.pixels.dtype == np.float32
    assert (first.total_pixels, first.valid_pixels) == (7, 5)
    assert first.detection == source.detections[0]
    assert result.metadata == source.metadata
    assert result.processed_path == source.processed_path
    assert result.json_path == tmp_path / "vari" / f"{Path(source.metadata.filename).stem}.json"
    record = json.loads(result.json_path.read_text())
    assert record["frame_id"] == 7
    assert record["captured_at"] == source.metadata.captured_at.isoformat()
    assert record["processed_path"] == str(source.processed_path)
    assert record["inference_json_path"] == str(source.json_path)
    assert record["npz_path"] == str(result.npz_path)
    with np.load(result.npz_path, allow_pickle=False) as arrays:
        assert arrays.files == ["detection_0", "detection_1"]
        for index, detection in enumerate(result.detections):
            entry = record["detections"][index]
            assert entry["detection_index"] == index
            assert entry["detection"]["class_name"] == f"plant_{index}"
            assert entry["bounds"] == list(detection.bounds)
            assert entry["valid_pixels"] == detection.valid_pixels
            np.testing.assert_allclose(arrays[entry["array_key"]], detection.pixels)


def test_grayscale_image_is_converted_to_rgb(tmp_path):
    source = make_result(tmp_path, [[[10, 30, 20]]], [(0, 0, 1, 1)])
    Image.new("L", (1, 1), color=50).save(source.processed_path)
    result = VariCalculator(tmp_path).compute(source)
    np.testing.assert_array_equal(result.detections[0].pixels, [[0]])
    assert result.detections[0].valid_pixels == 1


def test_fractional_and_clipped_bounds(tmp_path):
    source = make_result(tmp_path, [[[10, 30, 20]] * 3] * 2,
                         [(-0.2, 0.2, 1.2, 8), (1.2, -5, 9, 0.2)])
    result = VariCalculator(tmp_path).compute(source)
    assert [d.bounds for d in result.detections] == [(0, 0, 2, 2), (1, 0, 3, 1)]
    assert [d.pixels.shape for d in result.detections] == [(2, 2), (1, 2)]


@pytest.mark.parametrize("box,shape", [
    ((4, 0, 5, 1), (1, 0)), ((-5, 0, -1, 1), (1, 0)),
    ((0, 3, 1, 4), (0, 1)), ((0, 0, 0, 1), (1, 0)),
    ((0, 0, 1, 1), (1, 1)),
])
def test_empty_and_all_invalid_boxes(tmp_path, box, shape):
    source = make_result(tmp_path, [[[10, 20, 30]]], [box])
    result = VariCalculator(tmp_path).compute(source)
    detection = result.detections[0]
    assert detection.pixels.shape == shape
    assert detection.valid_pixels == 0
    raw = result.json_path.read_text()
    assert "NaN" not in raw


@pytest.mark.parametrize("box", [
    (float("nan"), 0, 1, 1), (0, float("inf"), 1, 1),
    (0, 0, float("-inf"), 1), (0, 0, 1, float("nan")),
    (2, 0, 1, 1), (0, 2, 1, 1),
])
def test_invalid_bounds_do_not_write_outputs(tmp_path, box):
    source = make_result(tmp_path, [[[1, 2, 3]]], [box])
    calculator = VariCalculator(tmp_path)
    with pytest.raises(ValueError, match="coordinates"):
        calculator.compute(source)
    assert list(calculator.vari_dir.iterdir()) == []


def test_recompute_replaces_outputs_with_empty_detections(tmp_path):
    source = make_result(tmp_path, [[[10, 30, 20]]], [(0, 0, 1, 1)])
    calculator = VariCalculator(tmp_path)
    initial = calculator.compute(source)
    result = calculator.compute(replace(source, detections=()))
    assert result.json_path == initial.json_path
    assert result.npz_path == initial.npz_path
    assert result.detections == ()
    assert json.loads(result.json_path.read_text())["detections"] == []
    with np.load(result.npz_path, allow_pickle=False) as arrays:
        assert arrays.files == []


def test_missing_image_and_wrong_input(tmp_path):
    source = make_result(tmp_path, [[[1, 2, 3]]], [])
    calculator = VariCalculator(tmp_path)
    with pytest.raises(FileNotFoundError):
        calculator.compute(replace(source, processed_path=tmp_path / "missing.png"))
    with pytest.raises(TypeError, match="InferenceResult"):
        calculator.compute(None)


def test_strict_json_rejects_nonfinite_metadata_before_writing(tmp_path):
    source = make_result(tmp_path, [[[10, 30, 20]]], [(0, 0, 1, 1)])
    source = replace(source, detections=(replace(source.detections[0], confidence=float("nan")),))
    calculator = VariCalculator(tmp_path)
    with pytest.raises(ValueError):
        calculator.compute(source)
    assert list(calculator.vari_dir.iterdir()) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
