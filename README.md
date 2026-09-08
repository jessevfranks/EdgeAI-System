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

Each experiment runs in a separate background process so the dashboard remains
usable. The app intentionally has no device scheduler, stop control, automatic
refresh, or interrupted-job recovery. Starting multiple experiments can run
them on the same GPU at the same time.

## Setup

Python 3.11 and a PyTorch build appropriate for the training computer are
required. From the repository root:

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
   scale, and enter a local Ultralytics dataset YAML.
2. On **Experiments**, use **Refresh** to update status and view stored metrics
   or the worker log.
3. For a completed tuning run, click **Train with best settings** to start a
   300-epoch run with its winning hyperparameters.
4. For a completed training run, choose `test` or `val` and click
   **Evaluate checkpoint**.

The form defaults are GPU `0`, image size `640`, batch `8`, 300 training
epochs, 20 epochs per tuning trial, and 10 tuning iterations.

Generated files are stored under `runs/`:

```text
runs/
|-- experiments.sqlite3
`-- <experiment-id>/
    |-- worker.log
    |-- train/
    |-- tune/
    `-- evaluate/
```

## Database reset

The simplified database is not compatible with the earlier experiment schema.
Before first launch, rename or remove the ignored `runs/experiments.sqlite3`
file and any matching `-wal` or `-shm` files. The app does not delete them
automatically.

## Code structure

```text
src/edge_ai/
|-- config.py       # Small experiment dataclass
|-- dashboard.py    # Two-page Streamlit UI and worker launcher
|-- experiments.py  # Ultralytics train, tune, and evaluate calls
|-- metrics.py      # Metric normalization and ingestion
|-- storage.py      # SQLite history
`-- worker.py       # One background experiment process
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
