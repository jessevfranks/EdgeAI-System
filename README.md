# EdgeAI System

EdgeAI System is a research project for fine-tuning and evaluating YOLOv8
object-detection models and, later, running selected models on a GPU-equipped
drone. The current implementation provides a local Streamlit workspace for
training experiments; it does not expose an experiment-management CLI.

## Implemented training workspace

The UI supports all five YOLOv8 detection scales:

- Nano (`yolov8n.pt`)
- Small (`yolov8s.pt`)
- Medium (`yolov8m.pt`)
- Large (`yolov8l.pt`)
- Extra large (`yolov8x.pt`)

Each experiment runs in an independent background process. From the UI you can
create, queue, stop, resume, clone, compare, promote, evaluate, and export
experiments. Closing the browser does not stop a running worker.

The model names above describe architecture scales, not numerical
quantization. TensorRT FP16 and INT8 export are intentionally deferred to the
deployment-optimization phase.

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

The launcher binds Streamlit to `localhost`. Training weights are downloaded by
Ultralytics when a scale is used for the first time.

## Dataset input

Experiments require a local Ultralytics dataset YAML. Training and validation
entries are mandatory. The UI requires a test entry when a test evaluation is
started and never silently substitutes the validation split.

```yaml
path: C:/datasets/my-dataset
train: images/train
val: images/val
test: images/test
names:
  0: target
```

An 80/10/10 split is the research starting point, but the application does not
move or split source data. It fingerprints the descriptor so the comparison
page can warn when experiments are not using equivalent datasets.

## Experiment profiles

[`configs/training.yaml`](configs/training.yaml) contains the editable defaults:

- `research_baseline`: SGD, learning rate 0.01, batch 8, 300 epochs, and
  patience 50.
- `starter_tune`: 10 Ultralytics genetic-tuning iterations with 20 epochs per
  trial.
- `final_train`: a 300-epoch run created explicitly from a tuning winner.

The starter tuning budget is useful for verifying the pipeline and exploring
directionally. It is too small to establish a final optimum for a real dataset.
The UI also exposes Ultralytics' default search space and a custom bounds editor.
Optimizers are compared as separate campaigns.

## UI workflow

1. Open **New Experiment**, select a dataset, model scale, and training or
   tuning profile, then create the experiment.
2. Use **Jobs** to monitor the queue, stop one worker, resume an interrupted
   run, clone a configuration, or inspect its log.
3. Use **Experiment Detail** for epoch curves, tuning trials, hyperparameters,
   progress, configuration, and artifacts.
4. Use **Compare** to inspect accuracy and fitness across compatible model
   scales. Differing dataset fingerprints are called out.
5. Use **Promote / Evaluate** to turn a completed tuning winner into an explicit
   full training job and evaluate a completed checkpoint on `test` or `val`.
6. Use **Exports** to download flattened CSV or structured JSON records.

Only one job is active on a given device string at a time. Other jobs for that
device remain queued. Completed tuning iterations are stored in NDJSON and
SQLite; if a worker is stopped during a trial, resuming re-runs only the
unfinished iteration.

## Storage and metrics

- `runs/experiments.sqlite3` uses two tables: one for experiment state and one
  flexible table for epoch, tuning, evaluation, and summary metrics.
- `runs/<experiment-id>/` stores immutable configuration manifests, worker
  logs, and raw Ultralytics outputs such as `results.csv` and
  `tune_results.ndjson`.
- `artifacts/models/<scale>/<experiment-id>/` stores explicitly promoted final
  checkpoints and their manifests.

Tracked metrics include precision, recall, F1, mAP50, mAP50-95, fitness,
training and validation losses, duration, validation speed, model metadata, and
peak CUDA memory when available. SQLite ingestion is idempotent, so dashboard
refreshes do not duplicate records.

Generated data, runs, databases, and weights are excluded from Git.

## How the code fits together

The training side intentionally uses a small design suitable for a research
project:

1. `ExperimentConfig` represents train, tune, and evaluation settings in one
   readable dataclass.
2. The dashboard saves that configuration in SQLite and asks `JobManager` to
   start it.
3. A separate worker calls the matching function in `experiments.py`, so a
   Streamlit refresh or closed browser does not stop training.
4. `metrics.py` reads Ultralytics' CSV or NDJSON files into generic JSON metric
   rows.
5. The dashboard reads those same rows for tables, charts, comparisons, and
   downloads.

There is no web API, ORM, migration framework, service daemon, or custom CLI.
The raw Ultralytics files remain available when more detailed investigation is
needed.

## Project structure

```text
EdgeAI-System/
|-- configs/                 # Training, evaluation, and onboard defaults
|-- scripts/
|   `-- run_dashboard.py     # Minimal localhost UI launcher
|-- src/edge_ai/
|   |-- config.py            # Typed experiment configuration
|   |-- dashboard.py         # Streamlit control surface
|   |-- experiments.py       # Training/tuning/evaluation operations
|   |-- jobs.py              # Worker queue and process control
|   |-- metrics.py           # Metric normalization and ingestion
|   |-- storage.py           # SQLite experiment registry
|   `-- worker.py            # Internal background worker protocol
|-- workflows/               # Thin offline workflow adapters
|-- tests/                    # Unit, integration, and opt-in GPU tests
|-- data/                     # Local datasets (ignored)
|-- artifacts/                # Promoted models (ignored)
`-- runs/                     # Experiment output and database (ignored)
```

The existing camera, preprocessing, inference, and onboard pipeline modules
remain reserved for the later drone-runtime implementation.

## Tests

```powershell
python -m pytest
```

The normal suite mocks Ultralytics and does not require a GPU. To run the
one-epoch COCO8 smoke test on a configured CUDA system:

```powershell
$env:EDGE_AI_GPU_SMOKE = "1"
python -m pytest -m gpu tests/integration/test_gpu_smoke.py
```
