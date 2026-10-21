from datetime import UTC
from pathlib import Path

import pytest
from PIL import Image

import onboard.sources as sources_module
from onboard import PacedImageDirectory


def _image(path: Path, color: str) -> None:
    Image.new("RGB", (2, 2), color).save(path)


def test_directory_source_preserves_filesystem_order(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "z.png"
    second = tmp_path / "a.jpg"
    ignored = tmp_path / "notes.txt"
    _image(first, "red")
    _image(second, "blue")
    ignored.write_text("not an image", encoding="utf-8")

    original_iterdir = Path.iterdir

    def ordered_iterdir(path: Path):
        if path == tmp_path:
            return iter((first, ignored, second))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", ordered_iterdir)
    monkeypatch.setattr(sources_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(sources_module.time, "sleep", lambda _delay: None)

    frames = list(PacedImageDirectory(tmp_path, fps=2))

    assert [frame.source_path for frame in frames] == [first, second]
    assert [frame.image.getpixel((0, 0)) for frame in frames] == [(255, 0, 0), (0, 0, 254)]
    assert all(frame.captured_at.tzinfo is UTC for frame in frames)


def test_directory_source_paces_from_initial_start(monkeypatch, tmp_path: Path) -> None:
    _image(tmp_path / "one.png", "red")
    _image(tmp_path / "two.png", "blue")
    times = iter((10.0, 10.0, 10.1))
    sleeps: list[float] = []
    monkeypatch.setattr(sources_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(sources_module.time, "sleep", sleeps.append)

    list(PacedImageDirectory(tmp_path, fps=2))

    assert sleeps == pytest.approx([0.4])


def test_directory_source_does_not_sleep_when_pipeline_is_behind(
    monkeypatch, tmp_path: Path
) -> None:
    _image(tmp_path / "one.png", "red")
    _image(tmp_path / "two.png", "blue")
    times = iter((0.0, 0.0, 2.0))
    sleeps: list[float] = []
    monkeypatch.setattr(sources_module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(sources_module.time, "sleep", sleeps.append)

    list(PacedImageDirectory(tmp_path, fps=2))

    assert sleeps == []


def test_directory_source_reports_corrupt_image(monkeypatch, tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.png"
    corrupt.write_text("broken", encoding="utf-8")
    monkeypatch.setattr(sources_module.time, "monotonic", lambda: 0.0)

    with pytest.raises(ValueError, match=r"Could not read image:.*broken\.png"):
        list(PacedImageDirectory(tmp_path, fps=1))


def test_directory_source_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        PacedImageDirectory(tmp_path / "missing", fps=1)

    image = tmp_path / "image.png"
    _image(image, "red")
    with pytest.raises(ValueError, match="must be a directory"):
        PacedImageDirectory(image, fps=1)
    with pytest.raises(ValueError, match="greater than zero"):
        PacedImageDirectory(tmp_path, fps=0)

    image.unlink()
    with pytest.raises(ValueError, match="contains no supported images"):
        PacedImageDirectory(tmp_path, fps=1)
