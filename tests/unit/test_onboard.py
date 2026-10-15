from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from onboard import ImageAcquirer, ImagePipeline


def test_acquire_stores_labeled_raw_rgb_image(tmp_path: Path) -> None:
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 128))
    captured_at = datetime(2026, 9, 10, 13, 23, 45, 123456, tzinfo=UTC)

    frame = ImageAcquirer(tmp_path).acquire(image, captured_at)

    expected_name = "frame_000001_20260910T132345123456Z.png"
    assert frame.metadata.frame_id == 1
    assert frame.metadata.captured_at == captured_at
    assert frame.raw_path == tmp_path / "raw" / expected_name
    assert not (tmp_path / "processed").exists()

    with Image.open(frame.raw_path) as raw:
        assert raw.mode == "RGB"
        assert list(raw.getdata()) == [(10, 20, 30)] * 4
        assert raw.info["frame_id"] == "1"
        assert raw.info["captured_at"] == captured_at.isoformat()


def test_capture_timestamp_is_converted_to_utc_or_added_at_receipt(
    tmp_path: Path,
) -> None:
    local_time = datetime(2026, 9, 10, 9, tzinfo=timezone(timedelta(hours=-4)))
    supplied = ImageAcquirer(tmp_path / "supplied").acquire(
        Image.new("RGB", (1, 1)), local_time
    )
    assert supplied.metadata.captured_at == datetime(2026, 9, 10, 13, tzinfo=UTC)

    before = datetime.now(UTC)
    received = ImageAcquirer(tmp_path / "received").acquire(Image.new("RGB", (1, 1)))
    after = datetime.now(UTC)
    assert before <= received.metadata.captured_at <= after
    assert received.metadata.captured_at.tzinfo is UTC


def test_invalid_image_and_timestamp_are_rejected(tmp_path: Path) -> None:
    acquirer = ImageAcquirer(tmp_path)
    with pytest.raises(TypeError, match="Pillow image"):
        acquirer.acquire(b"not an image")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone"):
        acquirer.acquire(
            Image.new("RGB", (1, 1)),
            datetime(2026, 9, 10),  # noqa: DTZ001
        )


def test_frame_sequence_resumes_from_existing_raw_files(tmp_path: Path) -> None:
    first = ImageAcquirer(tmp_path).acquire(
        Image.new("RGB", (1, 1)), datetime(2026, 9, 10, tzinfo=UTC)
    )
    assert first.metadata.frame_id == 1
    (tmp_path / "raw" / "unrelated.png").write_bytes(b"not a frame")
    (tmp_path / "raw" / "frame_999_backup.png").write_bytes(b"not a frame")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    (processed_dir / "frame_000004_20260910T120000000000Z.png").write_bytes(
        b"existing processed frame"
    )

    second = ImageAcquirer(tmp_path).acquire(
        Image.new("RGB", (1, 1)), datetime(2026, 9, 11, tzinfo=UTC)
    )
    assert second.metadata.frame_id == 2


def test_pipeline_runs_in_order_and_saves_labeled_rgb_image(tmp_path: Path) -> None:
    calls: list[str] = []

    def add_red(image: Image.Image) -> Image.Image:
        calls.append("red")
        red, green, blue = image.getpixel((0, 0))
        image.putpixel((0, 0), (red + 10, green, blue))
        return image

    def copy_red_to_green(image: Image.Image) -> Image.Image:
        calls.append("green")
        red, _, blue = image.getpixel((0, 0))
        image.putpixel((0, 0), (red, red, blue))
        return image

    acquired = ImageAcquirer(tmp_path).acquire(
        Image.new("RGB", (1, 1), (1, 2, 3)),
        datetime(2026, 9, 10, tzinfo=UTC),
    )
    processed = ImagePipeline(
        [add_red, copy_red_to_green], storage_root=tmp_path
    ).process(acquired)

    assert calls == ["red", "green"]
    assert processed.metadata == acquired.metadata
    assert (
        processed.processed_path == tmp_path / "processed" / acquired.metadata.filename
    )
    with (
        Image.open(acquired.raw_path) as raw,
        Image.open(processed.processed_path) as output,
    ):
        assert raw.getpixel((0, 0)) == (1, 2, 3)
        assert output.mode == "RGB"
        assert output.getpixel((0, 0)) == (11, 11, 3)
        assert output.info["frame_id"] == "1"
        assert output.info["captured_at"] == acquired.metadata.captured_at.isoformat()


@pytest.mark.parametrize(
    ("processor", "error", "message"),
    [
        (lambda image: None, TypeError, "must return a Pillow image"),
        (
            lambda image: (_ for _ in ()).throw(RuntimeError("failed")),
            RuntimeError,
            "failed",
        ),
    ],
)
def test_processing_failure_retains_raw_image_without_processed_output(
    tmp_path: Path, processor, error: type[Exception], message: str
) -> None:
    acquired = ImageAcquirer(tmp_path).acquire(Image.new("RGB", (1, 1)))
    pipeline = ImagePipeline([processor], storage_root=tmp_path)

    with pytest.raises(error, match=message):
        pipeline.process(acquired)

    assert acquired.raw_path.is_file()
    assert list(pipeline.processed_dir.iterdir()) == []


def test_pipeline_rejects_invalid_processor_and_frame(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be callable"):
        ImagePipeline([None], storage_root=tmp_path)  # type: ignore[list-item]

    pipeline = ImagePipeline(storage_root=tmp_path)
    with pytest.raises(TypeError, match="AcquiredFrame"):
        pipeline.process(None)  # type: ignore[arg-type]

    acquired = ImageAcquirer(tmp_path).acquire(Image.new("RGB", (1, 1)))
    acquired.raw_path.unlink()
    with pytest.raises(FileNotFoundError, match="Raw image"):
        pipeline.process(acquired)


def test_pipeline_converts_processor_output_to_rgb(tmp_path: Path) -> None:
    acquired = ImageAcquirer(tmp_path).acquire(Image.new("RGB", (1, 1)))
    pipeline = ImagePipeline(
        [lambda image: Image.new("L", image.size, 50)], storage_root=tmp_path
    )

    processed = pipeline.process(acquired)

    with Image.open(processed.processed_path) as output:
        assert output.mode == "RGB"
