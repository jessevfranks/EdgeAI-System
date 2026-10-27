import json
from pathlib import Path

import pytest

from onboard.metrics import FrameMetrics, MetricsCollector, SystemSample


def test_collector_writes_cpu_run_with_frame_timings(tmp_path: Path) -> None:
    collector = MetricsCollector(tmp_path, device="cpu", sampling_interval_seconds=60)

    with collector.run():
        collector.record_model_prediction(7, 3.0)
        collector.record_frame(7, pipeline_latency_ms=12.0, inference_stage_latency_ms=8.0)

    assert collector.last_output_path is not None
    record = json.loads(collector.last_output_path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["status"] == "completed"
    assert record["device"] == "cpu"
    assert record["frames"] == [
        {
            "frame_id": 7,
            "pipeline_latency_ms": 12.0,
            "inference_stage_latency_ms": 8.0,
            "model_prediction_latency_ms": 3.0,
        }
    ]
    assert record["summary"]["frame_count"] == 1
    assert record["summary"]["pipeline_latency_ms"] == {
        "mean": 12.0,
        "min": 12.0,
        "max": 12.0,
    }
    assert record["telemetry_availability"]["gpu_power"] == {
        "available": False,
        "reason": "Configured inference device is CPU",
    }
    assert "gpu_power_watts" not in record["summary"]
    assert all("gpu_power_watts" not in sample for sample in record["samples"])


def test_summary_integrates_gpu_power_samples() -> None:
    collector = MetricsCollector(device="cpu")
    collector._frames = [FrameMetrics(1, 9.0, 6.0, 4.0)]
    collector._samples = [
        SystemSample("start", 0.0, 10.0, 20.0, 30.0, 40.0),
        SystemSample("middle", 500.0, 20.0, 30.0, 50.0, 60.0),
        SystemSample("end", 1000.0, 30.0, 40.0, 70.0, 80.0),
    ]

    summary = collector._summary(run_duration_ms=1000.0)

    assert summary["gpu_power_watts"] == {
        "mean": 60.0,
        "min": 40.0,
        "max": 80.0,
    }
    assert summary["estimated_gpu_energy_joules"] == pytest.approx(60.0)


def test_collector_writes_interrupted_and_failed_runs(tmp_path: Path) -> None:
    interrupted = MetricsCollector(tmp_path, device="cpu", sampling_interval_seconds=60)
    with pytest.raises(KeyboardInterrupt), interrupted.run():
        raise KeyboardInterrupt

    failed = MetricsCollector(tmp_path, device="cpu", sampling_interval_seconds=60)
    with pytest.raises(RuntimeError, match="inference failed"), failed.run():
        raise RuntimeError("inference failed")

    interrupted_record = json.loads(
        interrupted.last_output_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    failed_record = json.loads(
        failed.last_output_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    assert interrupted_record["status"] == "interrupted"
    assert interrupted_record["error"]["type"] == "KeyboardInterrupt"
    assert failed_record["status"] == "failed"
    assert failed_record["error"] == {
        "type": "RuntimeError",
        "message": "inference failed",
    }
    assert interrupted.last_output_path != failed.last_output_path


def test_numeric_device_tolerates_unavailable_nvml(monkeypatch, tmp_path: Path) -> None:
    import onboard.metrics as metrics_module

    def unavailable() -> None:
        raise metrics_module.pynvml.NVMLError(
            metrics_module.pynvml.NVML_ERROR_LIBRARY_NOT_FOUND
        )

    monkeypatch.setattr(metrics_module.pynvml, "nvmlInit", unavailable)
    collector = MetricsCollector(tmp_path, device="0", sampling_interval_seconds=60)

    with collector.run():
        pass

    record = json.loads(
        collector.last_output_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    )
    availability = record["telemetry_availability"]
    assert availability["gpu_utilization"]["available"] is False
    assert availability["gpu_power"]["available"] is False
    assert "NVML unavailable" in availability["gpu_power"]["reason"]


def test_sampling_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        MetricsCollector(sampling_interval_seconds=0)
