from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from onboard import ImageAcquirer, ImagePipeline


def test_acquire_stores_labeled_raw_and_processed_rgb_images(tmp_path: Path) -> None:
    image = Image.new("RGBA", (2, 2), (10, 20, 30, 128))
    captured_at = datetime(2026, 9, 10, 13, 23, 45, 123456, tzinfo=UTC)

    frame = ImageAcquirer(tmp_path).acquire(image, captured_at)

    expected_name = "frame_000001_20260910T132345123456Z.png"
    assert frame.metadata.frame_id == 1
    assert frame.metadata.captured_at == captured_at
    assert frame.raw_path == tmp_path / "raw" / expected_name
    assert frame.processed_path == tmp_path / "processed" / expected_name

    with Image.open(frame.raw_path) as raw, Image.open(frame.processed_path) as processed:
        assert raw.mode == "RGB"
        assert list(raw.getdata()) == list(processed.getdata())
        for saved in (raw, processed):
            assert saved.info["frame_id"] == "1"
            assert saved.info["captured_at"] == captured_at.isoformat()


def test_capture_timestamp_is_converted_to_utc_or_added_at_receipt(tmp_path: Path) -> None:
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
        acquirer.acquire(Image.new("RGB", (1, 1)), datetime(2026, 9, 10))


def test_frame_sequence_resumes_from_existing_files(tmp_path: Path) -> None:
    first = ImageAcquirer(tmp_path).acquire(
        Image.new("RGB", (1, 1)), datetime(2026, 9, 10, tzinfo=UTC)
    )
    assert first.metadata.frame_id == 1
    (tmp_path / "raw" / "unrelated.png").write_bytes(b"not a frame")
    (tmp_path / "raw" / "frame_999_backup.png").write_bytes(b"not a frame")
    (tmp_path / "processed" / "frame_000004_20260910T120000000000Z.png").write_bytes(
        b"existing processed frame"
    )

    second = ImageAcquirer(tmp_path).acquire(
        Image.new("RGB", (1, 1)), datetime(2026, 9, 11, tzinfo=UTC)
    )
    assert second.metadata.frame_id == 5


def test_pipeline_runs_in_order_without_changing_raw_image(tmp_path: Path) -> None:
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

    pipeline = ImagePipeline()
    pipeline.add(add_red)
    pipeline.add(copy_red_to_green)
    frame = ImageAcquirer(tmp_path, pipeline).acquire(
        Image.new("RGB", (1, 1), (1, 2, 3)), datetime(2026, 9, 10, tzinfo=UTC)
    )

    assert calls == ["red", "green"]
    with Image.open(frame.raw_path) as raw, Image.open(frame.processed_path) as processed:
        assert raw.getpixel((0, 0)) == (1, 2, 3)
        assert processed.getpixel((0, 0)) == (11, 11, 3)


@pytest.mark.parametrize(
    ("processor", "error", "message"),
    [
        (lambda image: None, TypeError, "must return a Pillow image"),
        (lambda image: (_ for _ in ()).throw(RuntimeError("failed")), RuntimeError, "failed"),
    ],
)
def test_processing_failure_retains_raw_image(
    tmp_path: Path, processor, error: type[Exception], message: str
) -> None:
    acquirer = ImageAcquirer(tmp_path, ImagePipeline([processor]))

    with pytest.raises(error, match=message):
        acquirer.acquire(Image.new("RGB", (1, 1)), datetime(2026, 9, 10, tzinfo=UTC))

    assert len(list((tmp_path / "raw").glob("*.png"))) == 1
    assert list((tmp_path / "processed").glob("*.png")) == []


def test_pipeline_rejects_non_callable_processors() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        ImagePipeline([None])  # type: ignore[list-item]
