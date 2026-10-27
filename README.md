# EdgeAI System

EdgeAI System is a small school/research project for training and evaluating
YOLOv8 object-detection models from a local Streamlit dashboard.

## Features

- Train all five YOLOv8 model sizes: nano (`n`), small (`s`), medium (`m`),
  large (`l`), and extra-large (`x`).
- Tune hyperparameters with Ultralytics' built-in genetic algorithm.
- Evaluate trained checkpoints on the validation or test split.
- Store experiment status, epoch metrics, tuning trials, evaluation metrics,
  and winning hyperparameters in SQLite.
- Export the best checkpoint from CUDA training runs to ONNX and a static FP16
  TensorRT engine while retaining both PyTorch checkpoints.
- Run RGB camera frames through separate acquisition, processing, inference,
  and VARI stages while saving each stage's outputs.

Each experiment runs in a separate background process so the dashboard remains
usable. The app intentionally has no device scheduler, stop control, automatic
refresh, or interrupted-job recovery. Starting multiple experiments can run
them on the same GPU at the same time.

## Setup

Python 3.11 or newer and a PyTorch build appropriate for the training computer
are required. CUDA exports also require a compatible NVIDIA GPU, CUDA
installation, and TensorRT Python package on the machine that will run the
engine. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/run_dashboard.py
```

Ultralytics downloads model weights the first time a scale is used.

## Using the dashboard

1. On **New Experiment**, choose training or genetic tuning, select a YOLO
   scale, a CPU or detected GPU, and a local Ultralytics dataset YAML.
2. On **Experiments**, use **Refresh** to update status and view stored metrics
   or the worker log.
3. For a completed tuning run, click **Train with best settings** to start a
   300-epoch run with its winning hyperparameters.
4. For a completed training run, choose `test` or `val` and click
   **Evaluate checkpoint**.

The form defaults to CPU, image size `320`, batch `4`, one training epoch,
20 epochs per tuning trial, and 10 tuning iterations. If the local COCO8
dataset exists at `data/tuning/coco8/coco8.yaml`, the form selects it by
default. CPU runs use `device=cpu`; a detected GPU uses its numeric index,
such as `device=0`. Device choice is an experiment setting, not part of the
dataset YAML.

## Data and generated files

The ignored `data/` directory separates training datasets, repeatable test
datasets, and live onboard data. Every onboard pipeline stage writes beneath
the configured `storage_root`; its output directories are created as needed.

```text
data/
|-- tuning/
|   `-- <dataset>/                 # Dataset YAML, images, and labels
|-- testing/
|   `-- <dataset>/
|       |-- input/                 # Test images supplied by the user
|       |-- raw/                   # Stored RGB captures
|       |-- processed/             # Images after preprocessing
|       |-- annotated/             # Detection images
|       |-- detections/            # Detection JSON
|       |-- vari/                  # VARI JSON and NPZ arrays
|       |-- metrics/               # One runtime-metrics JSON per invocation
|       `-- vari_overlays/         # Optional colorized VARI images
`-- onboard/                       # Default live-pipeline storage_root
    |-- raw/
    |-- processed/
    |-- annotated/
    |-- detections/
    |-- vari/
    |-- metrics/
    `-- vari_overlays/                 # Optional
```

The dashboard keeps experiment history and Ultralytics run output under
`runs/`. After a training or tuning job completes, its PyTorch checkpoints are
moved to `artifacts/` under the same experiment ID. Training on CUDA also places
the ONNX and TensorRT exports beside `best.pt`:

```text
runs/
|-- experiments.sqlite3
`-- <experiment-id>/
    |-- worker.log
    `-- <train, tune, or evaluate>/
artifacts/
`-- <experiment-id>/
    `-- <train or tune>/weights/
        |-- best.pt
        |-- last.pt
        |-- best.onnx              # CUDA training only
        `-- best.engine            # CUDA training only
```

CPU training runs retain the `.pt` checkpoints and mark deployment export as
skipped. A TensorRT export error is shown separately in the experiment view and
does not discard or fail otherwise successful training. TensorRT engines are
specific to their build environment, so build and run an engine on a compatible
NVIDIA GPU and TensorRT version.

Raw, processed, and annotated images share a filename such as
`frame_000001_20260910T132345123456Z.png`; related JSON and NPZ files use the
same stem. Camera integrations provide a Pillow image to
`ImageAcquirer.acquire`. Pass its `AcquiredFrame` to `ImagePipeline.process`;
an empty processing pipeline saves an unchanged processed copy. Inference
consumes the resulting `ProcessedFrame`:

```python
from onboard import ImageAcquirer, ImagePipeline, InferencePipeline, VariCalculator

acquired = ImageAcquirer().acquire(camera_image)
processed = ImagePipeline().process(acquired)
inference = InferencePipeline(
    "artifacts/<experiment-id>/train/weights/best.engine",
    device="0",
).infer(processed)
vari = VariCalculator().compute(inference)

for detection in inference.detections:
    print(detection.class_name, detection.confidence)
```

`OnboardOrchestrator` constructs these stages and runs the same sequence for
applications that want a single entry point:

```python
from onboard import OnboardOrchestrator

result = OnboardOrchestrator(
    "artifacts/<experiment-id>/train/weights/best.engine",
    device="0",
).run(camera_image)
```

### Testing the onboard pipeline with images

For the included example, set `model`, `device`, `imgsz`, `confidence`, `iou`,
`storage_root`, and `fps` in `configs/testing.yaml`. Its checked-in `model`
value contains a machine-specific experiment ID, so replace it with an existing
local checkpoint before running the command:

```powershell
python scripts/run_onboard.py test-images data/testing/potted_plants/input `
    --config configs/testing.yaml
```

To also save a visual VARI heatmap over each annotated detection image, add
`--vari-overlay`:

```powershell
python scripts/run_onboard.py test-images data/testing/potted_plants/input `
    --config configs/testing.yaml --vari-overlay
```

The overlays are written to `<storage_root>/vari_overlays/`. Red represents low
VARI, green represents high VARI, and pixels with an undefined VARI are left
unchanged.

The runner examines only the top level of the directory. It processes PNG,
JPEG, AVIF, and WebP files once in the order returned by the operating system,
without sorting them. Other files are ignored. The first image is
processed immediately; later images are introduced at the configured FPS. Time
spent running the pipeline counts toward that interval, so the runner sleeps
only when it is ahead of schedule. It exits after all selected images finish,
or cleanly when interrupted with Ctrl+C.

Each image passes through acquisition, processing, inference, and VARI using
the same `storage_root`. The command prints frame completion and detection
counts and writes one timestamped runtime-metrics JSON file under
`<storage_root>/metrics/`.

Each metrics file includes per-frame pipeline, inference-stage, and model-call
latencies, along with periodic system and process CPU utilization. Numeric CUDA
devices also collect device-wide GPU utilization and power when NVML supports
those readings. The power samples are integrated into an estimated GPU energy
value. Missing GPU telemetry does not stop the pipeline; the file records why
it was unavailable. Metrics are finalized for completed, interrupted, and
failed runs.

For CPU testing, pass `best.pt` with `device="cpu"`. CUDA inference requires a
`.engine` model and a numeric device such as `"0"`; CPU inference requires a
`.pt` model. Inference does not export checkpoints or download model files.
Results use the frame filename and are stored under `<storage_root>/annotated/`
and `<storage_root>/detections/`.

### Per-detection VARI

Compute `(g-r)/(g+r-b)` from every pixel inside each detection's bounding box:

```python
import numpy as np
from onboard import VariCalculator

vari_result = VariCalculator().compute(inference)
for detection in vari_result.detections:
    print(detection.detection.class_name, detection.mean, detection.valid_pixels)

with np.load(vari_result.npz_path, allow_pickle=False) as arrays:
    first_map = arrays["detection_0"] if vari_result.detections else None
```

The calculator uses the processed RGB image so detection coordinates align.
Color-altering preprocessing therefore affects VARI. Box minima round down,
maxima round up, and bounds clip to the image; maximum bounds are exclusive.
Overlapping detections retain their own maps, including background pixels.
Zero denominators produce `NaN` pixels excluded from statistics; other values
are preserved, including values outside `[-1, 1]`.

Each frame produces `<storage_root>/vari/<frame_stem>.json` with detection
metadata, clipped bounds `(x_min, y_min, x_max, y_max)`, pixel counts, mean,
median, minimum, and maximum. A matching compressed `.npz` stores float32 maps
under `detection_0`, `detection_1`, etc., in inference order. JSON records each
array key and links to the source image, inference JSON, and NPZ file. Empty
boxes or boxes without valid pixels have null statistics. Frames without
detections produce an empty detection list and archive. Recomputing replaces
both files.

Keep the existing PNG images alongside these results: saved bounds reconstruct
RGB crops without storing duplicate images. Supply the same `storage_root` to
acquisition, processing, inference, and VARI calculation when using a custom
directory. The orchestrator does this for all stages automatically.

## Database migration

When the dashboard opens an earlier experiment database, it automatically
migrates the legacy experiment table to the current schema while preserving
configuration, history, run paths, errors, and stored metrics. No manual reset
is required.

## Code structure

```text
src/
|-- model_development/  # Training, evaluation, dashboard, and experiment history
`-- onboard/            # RGB acquisition, processing, inference, storage, and VARI
```

## Tests

The normal suite does not run the opt-in GPU smoke test:

```powershell
python -m pytest
```

The GPU smoke test remains opt-in:

```powershell
$env:EDGE_AI_GPU_SMOKE = "1"
python -m pytest -m gpu tests/integration/test_gpu_smoke.py
```
