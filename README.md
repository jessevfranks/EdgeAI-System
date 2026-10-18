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

Python 3.11 and a PyTorch build appropriate for the training computer are
required. CUDA exports also require a compatible NVIDIA GPU, CUDA installation,
and TensorRT Python package on the machine that will run the engine. From the
repository root:

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

Logs and metrics are stored under `runs/`. Completed training checkpoints are
moved into `artifacts/` under the same experiment ID:

```text
runs/
|-- experiments.sqlite3
`-- <experiment-id>/
    |-- worker.log
    |-- train/
    |-- tune/
    `-- evaluate/
artifacts/
`-- <experiment-id>/
    `-- train/weights/
        |-- best.pt
        |-- last.pt
        |-- best.onnx
        `-- best.engine
```

CPU training runs retain the `.pt` checkpoints and mark deployment export as
skipped. A TensorRT export error is shown separately in the experiment view and
does not discard or fail otherwise successful training. TensorRT engines are
specific to their build environment, so build and run an engine on a compatible
NVIDIA GPU and TensorRT version.

Onboard frames are stored under the ignored `data/onboard/` directory. Raw
captures and processed results share the same labeled PNG filename:

```text
data/onboard/
|-- raw/frame_000001_20260910T132345123456Z.png
`-- processed/frame_000001_20260910T132345123456Z.png
```

Camera integrations provide a Pillow image to `ImageAcquirer.acquire`. Pass its
`AcquiredFrame` to `ImagePipeline.process`; an empty processing pipeline saves
an unchanged processed copy. Inference consumes the resulting `ProcessedFrame`:

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

For CPU testing, pass `best.pt` with `device="cpu"`. CUDA inference requires a
`.engine` model and a numeric device such as `"0"`; CPU inference requires a
`.pt` model. Inference does not export checkpoints or download model files.
Results use the frame filename and are stored under `data/onboard/annotated/`
and `data/onboard/detections/`.

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

Each frame produces `data/onboard/vari/<frame_stem>.json` with detection metadata,
clipped bounds `(x_min, y_min, x_max, y_max)`, pixel counts, mean, median, minimum,
and maximum. A matching compressed `.npz` stores float32 maps under
`detection_0`, `detection_1`, etc., in inference order. JSON records each array
key and links to the source image, inference JSON, and NPZ file. Empty boxes
or boxes without valid pixels have null statistics. Frames without detections
produce an empty detection list and archive. Recomputing replaces both files.

Keep the existing PNG images alongside these results: saved bounds reconstruct
RGB crops without storing duplicate images. Supply the same `storage_root` to
acquisition, processing, inference, and VARI calculation when using a custom
directory. The orchestrator does this for all stages automatically.

## Database reset

The simplified database is not compatible with the earlier experiment schema.
Before first launch, rename or remove the ignored `runs/experiments.sqlite3`
file and any matching `-wal` or `-shm` files. The app does not delete them
automatically.

## Code structure

```text
src/
|-- model_development/  # Training, evaluation, dashboard, and experiment history
`-- onboard/            # RGB image acquisition, identity, storage, and processing
```

## Tests

The normal suite mocks Ultralytics and does not require a GPU:

```powershell
python -m pytest
```

The GPU smoke test remains opt-in:

```powershell
$env:EDGE_AI_GPU_SMOKE = "1"
python -m pytest -m gpu tests/integration/test_gpu_smoke.py
```
