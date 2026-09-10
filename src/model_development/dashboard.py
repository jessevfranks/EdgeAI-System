"""Small Streamlit UI for YOLOv8 experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st

from model_development.config import MODEL_SCALES, ExperimentConfig
from model_development.metrics import sync_metrics
from model_development.storage import ExperimentRecord, ExperimentStorage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "runs" / "experiments.sqlite3"


def _start_worker(storage: ExperimentStorage, record: ExperimentRecord) -> None:
    log_path = Path(record.run_dir) / "worker.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "edge_ai.worker",
        str(storage.path),
        record.id,
    ]
    try:
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
    except OSError as exc:
        storage.set_status(record.id, "failed", str(exc))
        raise


def _create_and_start(storage: ExperimentStorage, config: ExperimentConfig) -> ExperimentRecord:
    record = storage.create_experiment(config)
    _start_worker(storage, record)
    return record


def _metric_kind(record: ExperimentRecord) -> str:
    return {"train": "epoch", "tune": "trial", "evaluate": "evaluation"}[record.action]


def _best_fitness(rows: list[dict[str, Any]]) -> float | None:
    scores = [row["fitness"] for row in rows if isinstance(row.get("fitness"), (int, float))]
    return max(scores) if scores else None


def _promoted_config(record: ExperimentRecord, trial: dict[str, Any]) -> ExperimentConfig:
    source = ExperimentConfig.from_dict(record.config)
    return ExperimentConfig(
        action="train",
        name=f"{source.name} final",
        data=source.data,
        scale=source.scale,
        device=source.device,
        imgsz=source.imgsz,
        batch=source.batch,
        epochs=300,
        hyperparameters=trial.get("hyperparameters", {}),
    )


def _evaluation_config(
    record: ExperimentRecord, checkpoint: str, split: str
) -> ExperimentConfig:
    source = ExperimentConfig.from_dict(record.config)
    return ExperimentConfig(
        action="evaluate",
        name=f"{source.name} {split}",
        data=source.data,
        scale=source.scale,
        device=source.device,
        imgsz=source.imgsz,
        batch=source.batch,
        split=split,
        weights=checkpoint,
    )


def _new_experiment_page(storage: ExperimentStorage) -> None:
    st.header("New experiment")
    action = st.radio("Experiment type", ["train", "tune"], horizontal=True)

    with st.form("new-experiment"):
        name = st.text_input("Experiment name", value=f"YOLOv8n {action}")
        data = st.text_input("Dataset YAML", value=str(PROJECT_ROOT / "data" / "dataset.yaml"))
        left, right = st.columns(2)
        scale = left.selectbox("Model scale", MODEL_SCALES)
        device = right.text_input("Device", value="0")

        first, second, third = st.columns(3)
        imgsz = first.number_input("Image size", min_value=32, value=640)
        batch = second.number_input("Batch", min_value=1, value=8)
        default_epochs = 300 if action == "train" else 20
        epochs = third.number_input("Epochs", min_value=1, value=default_epochs)
        iterations = 10
        if action == "tune":
            iterations = st.number_input("Tuning iterations", min_value=1, value=10)
        submitted = st.form_submit_button("Start experiment", type="primary")

    if not submitted:
        return

    try:
        config = ExperimentConfig(
            action=action,
            name=name,
            data=str(Path(data).expanduser().resolve()),
            scale=scale,
            device=device,
            imgsz=int(imgsz),
            batch=int(batch),
            epochs=int(epochs),
            iterations=int(iterations),
        )
        record = _create_and_start(storage, config)
        st.success(f"Started {record.name} ({record.id})")
    except Exception as exc:  # noqa: BLE001 - show input and launch errors in the UI
        st.error(str(exc))


def _show_metrics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No metrics have been written yet.")
        return
    st.dataframe(rows, hide_index=True, use_container_width=True)
    chart_columns = [
        key
        for key in ("fitness", "map50_95", "map50", "precision", "recall")
        if any(key in row for row in rows)
    ]
    if len(rows) > 1 and chart_columns:
        st.line_chart(rows, x="step", y=chart_columns)


def _experiment_actions(
    storage: ExperimentStorage, record: ExperimentRecord, rows: list[dict[str, Any]]
) -> None:
    if record.status != "completed":
        return

    if record.action == "tune":
        scored = [row for row in rows if isinstance(row.get("fitness"), (int, float))]
        if scored and st.button("Train with best settings"):
            best = max(scored, key=lambda row: row["fitness"])
            try:
                promoted = _create_and_start(storage, _promoted_config(record, best))
                st.success(f"Started {promoted.name}")
            except Exception as exc:  # noqa: BLE001 - show workflow errors in the UI
                st.error(str(exc))

    if record.action == "train" and record.best_checkpoint:
        split = st.radio("Evaluation split", ["test", "val"], horizontal=True)
        if st.button("Evaluate checkpoint"):
            try:
                config = _evaluation_config(record, record.best_checkpoint, split)
                evaluation = _create_and_start(storage, config)
                st.success(f"Started {evaluation.name}")
            except Exception as exc:  # noqa: BLE001 - show workflow errors in the UI
                st.error(str(exc))


def _experiments_page(storage: ExperimentStorage) -> None:
    st.header("Experiments")
    records = storage.list()
    if not records:
        st.info("No experiments have been created yet.")
        return

    table = []
    for record in records:
        rows = storage.metrics(record.id, _metric_kind(record))
        table.append(
            {
                "name": record.name,
                "type": record.action,
                "scale": record.scale,
                "status": record.status,
                "device": record.device,
                "progress": len(rows),
                "best fitness": _best_fitness(rows),
                "created": record.created_at,
            }
        )
    st.dataframe(table, hide_index=True, use_container_width=True)

    labels = {record.id: f"{record.name} · {record.scale} · {record.status}" for record in records}
    selected_id = st.selectbox("Experiment", list(labels), format_func=labels.get)
    record = storage.get(selected_id)
    if record.status in {"starting", "running"}:
        record = sync_metrics(storage, record)
    rows = storage.metrics(record.id, _metric_kind(record))

    st.subheader(record.name)
    st.caption(f"Status: {record.status} · Model: YOLOv8{record.scale} · Device: {record.device}")
    if record.error:
        st.error(record.error)
    _show_metrics(rows)
    _experiment_actions(storage, record, rows)

    with st.expander("Configuration"):
        st.json(record.config)
    log_path = Path(record.run_dir) / "worker.log"
    if log_path.exists():
        with st.expander("Worker log"):
            st.code(log_path.read_text(encoding="utf-8", errors="replace")[-12000:])


def main() -> None:
    st.set_page_config(page_title="EdgeAI Experiments", page_icon="🎯", layout="wide")
    st.title("EdgeAI YOLOv8 Experiments")
    storage = ExperimentStorage(DATABASE_PATH)
    page = st.sidebar.radio("Workspace", ["New Experiment", "Experiments"])
    st.sidebar.button("Refresh")

    if page == "New Experiment":
        _new_experiment_page(storage)
    else:
        _experiments_page(storage)


if __name__ == "__main__":
    main()
