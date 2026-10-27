"""Collect latency and system telemetry for an onboard pipeline run."""

from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pynvml


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    frame_id: int
    pipeline_latency_ms: float
    inference_stage_latency_ms: float
    model_prediction_latency_ms: float


@dataclass(frozen=True, slots=True)
class SystemSample:
    captured_at: str
    elapsed_ms: float
    system_cpu_percent: float
    process_cpu_percent: float
    gpu_utilization_percent: float | None = None
    gpu_power_watts: float | None = None


class MetricsCollector:
    """Collect one JSON record for each explicitly started pipeline run."""

    def __init__(
        self,
        storage_root: str | Path = "data/onboard",
        device: str = "0",
        sampling_interval_seconds: float = 0.25,
    ) -> None:
        if sampling_interval_seconds <= 0:
            raise ValueError("Metrics sampling interval must be greater than zero")

        self.metrics_dir = Path(storage_root).expanduser().resolve() / "metrics"
        self.device = device.strip().lower()
        self.sampling_interval_seconds = sampling_interval_seconds
        self.last_output_path: Path | None = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._gpu_handle: object | None = None
        self._nvml_initialized = False
        self._active = False
        self._started_at = datetime.now(UTC)
        self._started_counter = 0.0
        self._samples: list[SystemSample] = []
        self._frames: list[FrameMetrics] = []
        self._model_latencies: dict[int, float] = {}
        self._availability: dict[str, dict[str, object]] = {}

    @contextmanager
    def run(self) -> Iterator[None]:
        """Collect and persist one run, including interrupted and failed runs."""
        self.start()
        try:
            yield
        except KeyboardInterrupt as error:
            self._finish_after_error("interrupted", error)
            raise
        except BaseException as error:
            self._finish_after_error("failed", error)
            raise
        else:
            self.finish("completed")

    def start(self) -> None:
        if self._active:
            raise RuntimeError("Metrics collection is already active")

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._samples = []
        self._frames = []
        self._model_latencies = {}
        self._availability = {
            "system_cpu": {"available": True},
            "process_cpu": {"available": True},
        }
        self._stop_event.clear()
        self._started_at = datetime.now(UTC)
        self._started_counter = time.perf_counter()
        self.last_output_path = None
        self._initialize_gpu()

        # Prime psutil's non-blocking percentage counters before sampling.
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        self._active = True
        self._capture_sample()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="onboard-metrics",
            daemon=True,
        )
        self._thread.start()

    def finish(
        self,
        status: str,
        error: BaseException | None = None,
    ) -> Path:
        if not self._active:
            raise RuntimeError("Metrics collection is not active")
        if status not in {"completed", "interrupted", "failed"}:
            raise ValueError(f"Unknown metrics run status: {status}")

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._capture_sample()
        finished_counter = time.perf_counter()
        finished_at = datetime.now(UTC)
        self._shutdown_gpu()
        self._active = False

        run_id = (
            self._started_at.strftime("%Y%m%dT%H%M%S%fZ")
            + f"_{uuid.uuid4().hex[:8]}"
        )
        output_path = self.metrics_dir / f"run_{run_id}.json"
        record: dict[str, object] = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "status": status,
            "device": self.device,
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "telemetry_availability": self._availability,
            "summary": self._summary(
                run_duration_ms=(finished_counter - self._started_counter) * 1000
            ),
            "frames": [asdict(frame) for frame in self._frames],
            "samples": [self._sample_record(sample) for sample in self._samples],
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        output_path.write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self.last_output_path = output_path
        return output_path

    def record_model_prediction(self, frame_id: int, latency_ms: float) -> None:
        if not self._active:
            return
        with self._lock:
            self._model_latencies[frame_id] = latency_ms

    def record_frame(
        self,
        frame_id: int,
        pipeline_latency_ms: float,
        inference_stage_latency_ms: float,
    ) -> None:
        if not self._active:
            return
        with self._lock:
            model_latency = self._model_latencies.pop(frame_id)
            self._frames.append(
                FrameMetrics(
                    frame_id=frame_id,
                    pipeline_latency_ms=pipeline_latency_ms,
                    inference_stage_latency_ms=inference_stage_latency_ms,
                    model_prediction_latency_ms=model_latency,
                )
            )

    def _finish_after_error(self, status: str, error: BaseException) -> None:
        try:
            self.finish(status, error)
        except OSError:
            # Do not replace the pipeline exception with a metrics write failure.
            pass

    def _initialize_gpu(self) -> None:
        self._gpu_handle = None
        self._nvml_initialized = False
        if not self.device.isdecimal():
            reason = "Configured inference device is CPU"
            self._availability["gpu_utilization"] = {
                "available": False,
                "reason": reason,
            }
            self._availability["gpu_power"] = {
                "available": False,
                "reason": reason,
            }
            return

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device))
        except pynvml.NVMLError as error:
            reason = f"NVML unavailable: {error}"
            self._availability["gpu_utilization"] = {
                "available": False,
                "reason": reason,
            }
            self._availability["gpu_power"] = {
                "available": False,
                "reason": reason,
            }
            self._shutdown_gpu()
            return

        self._availability["gpu_utilization"] = {"available": True}
        self._availability["gpu_power"] = {"available": True}

    def _shutdown_gpu(self) -> None:
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
        self._nvml_initialized = False
        self._gpu_handle = None

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self.sampling_interval_seconds):
            self._capture_sample()

    def _capture_sample(self) -> None:
        captured_at = datetime.now(UTC).isoformat()
        elapsed_ms = (time.perf_counter() - self._started_counter) * 1000
        sample = SystemSample(
            captured_at=captured_at,
            elapsed_ms=elapsed_ms,
            system_cpu_percent=psutil.cpu_percent(interval=None),
            process_cpu_percent=self._process.cpu_percent(interval=None),
            gpu_utilization_percent=self._read_gpu_utilization(),
            gpu_power_watts=self._read_gpu_power(),
        )
        with self._lock:
            self._samples.append(sample)

    def _read_gpu_utilization(self) -> float | None:
        if self._gpu_handle is None:
            return None
        try:
            return float(pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu)
        except pynvml.NVMLError as error:
            self._mark_gpu_unavailable("gpu_utilization", error)
            return None

    def _read_gpu_power(self) -> float | None:
        if self._gpu_handle is None:
            return None
        try:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
            return float(milliwatts) / 1000
        except pynvml.NVMLError as error:
            self._mark_gpu_unavailable("gpu_power", error)
            return None

    def _mark_gpu_unavailable(self, name: str, error: Exception) -> None:
        with self._lock:
            self._availability[name] = {
                "available": False,
                "reason": f"NVML reading unavailable: {error}",
            }

    def _summary(self, run_duration_ms: float) -> dict[str, object]:
        summary: dict[str, object] = {
            "frame_count": len(self._frames),
            "run_duration_ms": run_duration_ms,
        }
        for field in (
            "pipeline_latency_ms",
            "inference_stage_latency_ms",
            "model_prediction_latency_ms",
        ):
            values = [getattr(frame, field) for frame in self._frames]
            if values:
                summary[field] = self._statistics(values)

        for field in (
            "system_cpu_percent",
            "process_cpu_percent",
            "gpu_utilization_percent",
            "gpu_power_watts",
        ):
            values = [
                value
                for sample in self._samples
                if (value := getattr(sample, field)) is not None
            ]
            if values:
                summary[field] = self._statistics(values)

        energy = 0.0
        has_power_interval = False
        for first, second in zip(self._samples, self._samples[1:]):
            if first.gpu_power_watts is None or second.gpu_power_watts is None:
                continue
            elapsed_seconds = (second.elapsed_ms - first.elapsed_ms) / 1000
            average_watts = (
                first.gpu_power_watts + second.gpu_power_watts
            ) / 2
            energy += average_watts * elapsed_seconds
            has_power_interval = True
        if has_power_interval:
            summary["estimated_gpu_energy_joules"] = energy
        return summary

    @staticmethod
    def _statistics(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }

    @staticmethod
    def _sample_record(sample: SystemSample) -> dict[str, object]:
        record = asdict(sample)
        return {key: value for key, value in record.items() if value is not None}
