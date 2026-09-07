"""Local Streamlit control surface for all YOLOv8 experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from edge_ai.config import (
    CORE_WIDE_SEARCH_SPACE,
    ExperimentConfig,
    ModelScale,
    load_dataset_yaml,
    load_yaml,
    merged_profile,
)
from edge_ai.experiments import create_evaluation, promote_best
from edge_ai.jobs import JobManager
from edge_ai.metrics import sync_metrics
from edge_ai.storage import ExperimentRecord, ExperimentStorage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "configs" / "training.yaml"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@st.cache_data(show_spinner=False)
def _settings() -> dict[str, Any]:
    return load_yaml(SETTINGS_PATH)


def _services() -> tuple[ExperimentStorage, JobManager]:
    database = _project_path(_settings().get("paths", {}).get("database", "runs/experiments.sqlite3"))
    storage = ExperimentStorage(database)
    manager = JobManager(database)
    manager.reconcile()
    return storage, manager


def _labels(records: list[ExperimentRecord]) -> dict[str, str]:
    return {
        record.id: f"{record.name} · {record.scale} · {record.action} · {record.status} · {record.id}"
        for record in records
    }


def _config_paths() -> dict[str, Path]:
    paths = _settings().get("paths", {})
    return {
        "database_path": _project_path(paths.get("database", "runs/experiments.sqlite3")),
        "run_root": _project_path(paths.get("run_root", "runs")),
        "artifact_root": _project_path(paths.get("artifact_root", "artifacts")),
    }


def _create_experiment_page(storage: ExperimentStorage, manager: JobManager) -> None:
    st.header("New experiment")
    st.caption("Create one independent YOLOv8 scale experiment. Busy devices queue jobs automatically.")
    settings = _settings()
    mode_label = st.radio("Experiment type", ["Train baseline", "Tune hyperparameters"], horizontal=True)
    mode = "tune" if mode_label.startswith("Tune") else "train"
    profiles = {
        name: values
        for name, values in settings.get("profiles", {}).items()
        if values.get("mode", "train") == mode and name != "final_train"
    }
    profile_name = st.selectbox("Profile", list(profiles) or ["starter_tune" if mode == "tune" else "research_baseline"])
    profile = merged_profile(settings, profile_name)
    default_name = f"YOLOv8{ModelScale.NANO.value} {profile_name.replace('_', ' ')}"

    with st.form("new-experiment"):
        name = st.text_input("Experiment name", value=default_name)
        data = st.text_input("Dataset YAML", value=str(PROJECT_ROOT / "data" / "dataset.yaml"))
        left, middle, right = st.columns(3)
        scale = left.selectbox("Model scale", [item.value for item in ModelScale])
        optimizer_options = ["SGD", "AdamW", "Adam", "NAdam", "RAdam", "RMSProp"]
        if mode == "train":
            optimizer_options.append("auto")
        optimizer = middle.selectbox("Optimizer", optimizer_options, index=0)
        device = right.text_input("Device", value=str(profile.get("device", "0")))

        col1, col2, col3, col4 = st.columns(4)
        imgsz = col1.number_input("Image size", min_value=32, value=int(profile.get("imgsz", 640)), step=32)
        batch = col2.number_input("Batch", min_value=1, value=int(profile.get("batch", 8)), step=1)
        epochs = col3.number_input("Epochs per run", min_value=1, value=int(profile.get("epochs", 20)), step=1)
        patience = col4.number_input("Patience", min_value=0, value=int(profile.get("patience", 50)), step=1)

        workers = st.number_input("Data-loader workers", min_value=0, value=int(profile.get("workers", 8)), step=1)
        seed = st.number_input("Random seed", min_value=0, value=int(profile.get("seed", 42)), step=1)
        amp = st.checkbox("Automatic mixed precision", value=bool(profile.get("amp", True)))
        cache = st.checkbox("Cache dataset", value=bool(profile.get("cache", False)))

        iterations = int(profile.get("iterations", 10))
        search_profile = "core_wide"
        custom_space: dict[str, tuple[float, float]] = {}
        if mode == "tune":
            iterations = st.number_input("Tuning iterations", min_value=1, value=iterations, step=1)
            search_profile = st.selectbox("Search space", ["core_wide", "ultralytics_default", "custom"])
            if search_profile == "custom":
                configured_space = settings.get("search_spaces", {}).get(
                    "core_wide", CORE_WIDE_SEARCH_SPACE
                )
                bounds = pd.DataFrame(
                    [
                        {"parameter": key, "minimum": value[0], "maximum": value[1]}
                        for key, value in configured_space.items()
                    ]
                )
                edited = st.data_editor(
                    bounds, num_rows="dynamic", use_container_width=True, key="custom-bounds"
                )
                for row in edited.to_dict("records"):
                    if row.get("parameter"):
                        custom_space[str(row["parameter"])] = (float(row["minimum"]), float(row["maximum"]))

        submitted = st.form_submit_button("Create and start experiment", type="primary")

    if not submitted:
        return
    try:
        dataset = _project_path(data)
        load_dataset_yaml(dataset)
        common: dict[str, Any] = {
            "action": mode,
            "data": str(dataset),
            "scale": ModelScale.parse(scale),
            "name": name,
            "profile": profile_name,
            "optimizer": optimizer,
            "device": device,
            "imgsz": int(imgsz),
            "batch": int(batch),
            "epochs": int(epochs),
            "patience": int(patience),
            "seed": int(seed),
            "workers": int(workers),
            "cache": cache,
            "amp": amp,
            **_config_paths(),
        }
        if mode == "tune":
            config = ExperimentConfig(
                **common,
                iterations=int(iterations),
                search_profile=search_profile,
                search_space=custom_space,
            )
        else:
            extra = {key: value for key, value in profile.items() if key in {"lr0", "lrf", "momentum", "weight_decay"}}
            config = ExperimentConfig(**common, extra_args=extra)
        record = storage.create_experiment(config)
        record = manager.enqueue(record.id)
        st.success(f"Created {record.name} ({record.id}); status: {record.status}")
    except Exception as exc:  # noqa: BLE001 - UI boundary must report backend errors
        st.error(str(exc))


def _jobs_page(storage: ExperimentStorage, manager: JobManager) -> None:
    st.header("Jobs")
    records = storage.list()
    if not records:
        st.info("No experiments have been created yet.")
        return
    rows = [
        {
            "id": record.id,
            "name": record.name,
            "action": record.action,
            "scale": record.scale,
            "status": record.status,
            "device": record.device,
            "progress": f"{record.current_iteration}/{record.total_iterations or '—'}",
            "best fitness": record.best_fitness,
            "started": record.started_at,
        }
        for record in records
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    labels = _labels(records)
    selected_id = st.selectbox("Selected job", list(labels), format_func=labels.get)
    selected = storage.get(selected_id)
    stop_col, resume_col, clone_col = st.columns(3)
    if stop_col.button(
        "Stop", disabled=selected.status not in {"queued", "running"}, use_container_width=True
    ):
        manager.stop(selected.id)
        st.rerun()
    if resume_col.button(
        "Resume", disabled=selected.status not in {"stopped", "interrupted", "failed"}, use_container_width=True
    ):
        manager.resume(selected.id)
        st.rerun()
    if clone_col.button("Clone", use_container_width=True):
        clone = storage.clone(selected.id)
        manager.enqueue(clone.id)
        st.success(f"Cloned as {clone.id}")

    if selected.error:
        st.error(selected.error)
    log_path = Path(selected.run_dir) / "worker.log"
    if log_path.exists():
        with st.expander("Worker log", expanded=selected.status in {"failed", "interrupted"}):
            st.code(log_path.read_text(encoding="utf-8", errors="replace")[-12000:], language="text")


def _detail_page(storage: ExperimentStorage) -> None:
    st.header("Experiment detail")
    records = storage.list()
    if not records:
        st.info("No experiments have been created yet.")
        return
    labels = _labels(records)
    selected_id = st.selectbox("Experiment", list(labels), format_func=labels.get, key="detail-experiment")
    record = sync_metrics(storage, storage.get(selected_id))
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Status", record.status)
    metric2.metric("Scale", record.scale)
    metric3.metric("Best fitness", "—" if record.best_fitness is None else f"{record.best_fitness:.5f}")
    metric4.metric("Progress", f"{record.current_iteration}/{record.total_iterations or '—'}")

    st.subheader("Configuration")
    st.json(record.config)
    manifest = Path(record.run_dir) / "manifest.json"
    if manifest.exists():
        with st.expander("Environment"):
            st.json(json.loads(manifest.read_text(encoding="utf-8"))["environment"])
    summary = storage.metrics(record.id, "summary")
    if summary:
        st.subheader("Run summary")
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    if record.action == "tune":
        trials = storage.metrics(record.id, "trial")
        if trials:
            table = pd.json_normalize(trials, sep=".").rename(columns={"step": "iteration"})
            st.subheader("Tuning trials")
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.line_chart(table.set_index("iteration")[["fitness"]])
            hyperparameter_columns = [column for column in table.columns if column.startswith("hyperparameters.")]
            if hyperparameter_columns:
                parameter = st.selectbox("Hyperparameter vs. fitness", hyperparameter_columns)
                st.scatter_chart(table, x=parameter, y="fitness")
    elif record.action == "train":
        epochs = storage.metrics(record.id, "epoch")
        if epochs:
            table = pd.DataFrame(epochs).rename(columns={"step": "epoch"})
            st.subheader("Epoch metrics")
            st.dataframe(table, hide_index=True, use_container_width=True)
            chart_columns = [key for key in ("map50_95", "map50", "precision", "recall", "fitness") if key in table]
            if chart_columns:
                st.line_chart(table.set_index("epoch")[chart_columns])
            loss_columns = [key for key in table if "loss" in key]
            if loss_columns:
                st.line_chart(table.set_index("epoch")[loss_columns])
    else:
        evaluations = storage.metrics(record.id, "evaluation")
        if evaluations:
            st.subheader(f"{record.config['split'].title()} metrics")
            st.dataframe(pd.DataFrame(evaluations), hide_index=True, use_container_width=True)

    artifacts = storage.artifacts(record.id)
    if artifacts:
        st.subheader("Artifacts")
        st.dataframe(pd.DataFrame(artifacts), hide_index=True, use_container_width=True)


def _compare_page(storage: ExperimentStorage) -> None:
    st.header("Compare experiments")
    records = storage.list()
    if not records:
        st.info("No results are available yet.")
        return
    scale_filter = st.multiselect("Model scales", [item.value for item in ModelScale], default=[item.value for item in ModelScale])
    action_filter = st.multiselect("Experiment types", ["train", "tune", "evaluate"], default=["train", "tune", "evaluate"])
    selected = [record for record in records if record.scale in scale_filter and record.action in action_filter]
    fingerprints = {record.dataset_fingerprint for record in selected}
    if len(fingerprints) > 1:
        st.warning("These experiments use different dataset fingerprints and are not directly comparable.")
    rows = storage.export(selected)
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
    if not table.empty and table["best_fitness"].notna().any():
        chart = table.loc[table["best_fitness"].notna(), ["name", "best_fitness"]].set_index("name")
        st.bar_chart(chart)


def _promote_page(storage: ExperimentStorage, manager: JobManager) -> None:
    st.header("Promote and evaluate")
    tunes = storage.list(statuses=["completed"], actions=["tune"])
    st.subheader("Promote a tuning winner")
    if tunes:
        labels = _labels(tunes)
        tune_id = st.selectbox("Completed tuning experiment", list(labels), format_func=labels.get)
        trials = storage.metrics(tune_id, "trial")
        scored_trials = [trial for trial in trials if isinstance(trial.get("fitness"), (int, float))]
        if scored_trials:
            best = max(scored_trials, key=lambda item: item["fitness"])
            st.metric("Winning fitness", f"{best['fitness']:.5f}")
            st.json(best["hyperparameters"])
        else:
            st.warning("This tuning experiment does not have a scored trial to promote.")
        if st.button(
            "Promote and start 300-epoch training", type="primary", disabled=not scored_trials
        ):
            try:
                promoted = promote_best(tune_id, storage.path)
                manager.enqueue(promoted.id)
                st.success(f"Created final training experiment {promoted.id}")
            except Exception as exc:  # noqa: BLE001 - UI boundary must report backend errors
                st.error(str(exc))
    else:
        st.info("Complete a tuning experiment before promoting its best trial.")

    st.divider()
    st.subheader("Evaluate a trained checkpoint")
    trained = [
        record
        for record in storage.list(statuses=["completed"], actions=["train"])
        if record.best_checkpoint
    ]
    if trained:
        labels = _labels(trained)
        source_id = st.selectbox("Completed training experiment", list(labels), format_func=labels.get)
        split = st.radio("Evaluation split", ["test", "val"], horizontal=True)
        if st.button("Start evaluation"):
            try:
                evaluation = create_evaluation(source_id, storage.path, split=split)
                manager.enqueue(evaluation.id)
                st.success(f"Created evaluation {evaluation.id}")
            except Exception as exc:  # noqa: BLE001 - UI boundary must report backend errors
                st.error(str(exc))
    else:
        st.info("No completed training checkpoint is available.")


def _exports_page(storage: ExperimentStorage) -> None:
    st.header("Export results")
    records = storage.list()
    statuses = sorted({record.status for record in records})
    selected_statuses = st.multiselect("Statuses", statuses, default=statuses)
    selected = [record for record in records if record.status in selected_statuses]
    rows = storage.export(selected)
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    csv_data = frame.to_csv(index=False).encode("utf-8")
    json_rows = []
    for record, summary in zip(selected, rows):
        history = {}
        for kind in ("epoch", "trial", "evaluation", "summary"):
            if values := storage.metrics(record.id, kind):
                history[kind] = values
        json_rows.append(
            {
                "summary": summary,
                "config": record.config,
                "metrics": history,
            }
        )
    json_data = json.dumps(json_rows, indent=2, default=str).encode("utf-8")
    csv_col, json_col = st.columns(2)
    csv_col.download_button("Download CSV", csv_data, "edge_ai_experiments.csv", "text/csv", use_container_width=True)
    json_col.download_button(
        "Download JSON", json_data, "edge_ai_experiments.json", "application/json", use_container_width=True
    )


def main() -> None:
    st.set_page_config(page_title="EdgeAI Experiments", page_icon="🎯", layout="wide")
    st.title("EdgeAI YOLOv8 Experiments")
    storage, manager = _services()
    with st.sidebar:
        page = st.radio(
            "Workspace",
            ["New Experiment", "Jobs", "Experiment Detail", "Compare", "Promote / Evaluate", "Exports"],
        )
        live = st.toggle("Live refresh", value=page in {"Jobs", "Experiment Detail"})
        if live:
            st_autorefresh(interval=5000, key=f"refresh-{page}")
        st.caption(f"Database: {storage.path}")

    pages = {
        "New Experiment": lambda: _create_experiment_page(storage, manager),
        "Jobs": lambda: _jobs_page(storage, manager),
        "Experiment Detail": lambda: _detail_page(storage),
        "Compare": lambda: _compare_page(storage),
        "Promote / Evaluate": lambda: _promote_page(storage, manager),
        "Exports": lambda: _exports_page(storage),
    }
    pages[page]()


if __name__ == "__main__":
    main()
