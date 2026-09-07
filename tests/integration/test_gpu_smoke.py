import os

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("EDGE_AI_GPU_SMOKE") != "1", reason="opt-in GPU/network smoke test")
def test_ultralytics_coco8_smoke() -> None:
    from ultralytics import YOLO

    result = YOLO("yolov8n.pt").train(data="coco8.yaml", epochs=1, imgsz=320, batch=2, device=0)
    assert result is not None
