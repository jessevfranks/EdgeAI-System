from pathlib import Path

import pytest

from model_development.metrics import normalize_metrics, read_results_csv, read_tune_ndjson


def test_normalize_detection_metrics_and_fitness() -> None:
    metrics = normalize_metrics(
        {
            "metrics/precision(B)": "0.8",
            "metrics/recall(B)": "0.5",
            "metrics/mAP50(B)": "0.7",
            "metrics/mAP50-95(B)": "0.4",
        }
    )
    assert metrics["precision"] == 0.8
    assert metrics["f1"] == 2 * 0.8 * 0.5 / 1.3
    assert metrics["fitness"] == pytest.approx(0.43)


def test_results_csv_tolerates_headers_and_rows(tmp_path: Path) -> None:
    source = tmp_path / "results.csv"
    source.write_text(
        " epoch, metrics/precision(B), metrics/mAP50-95(B)\n1,0.7,0.4\n2,0.8,0.5\n",
        encoding="utf-8",
    )
    rows = read_results_csv(source)
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[-1]["map50_95"] == 0.5


def test_ndjson_skips_partial_line_and_keeps_complete_trials(tmp_path: Path) -> None:
    source = tmp_path / "tune_results.ndjson"
    source.write_text(
        '{"iteration":1,"fitness":0.4,"hyperparameters":{"lr0":0.01},'
        '"datasets":{"data":{"metrics/mAP50-95(B)":0.4}}}\n{"iteration":2',
        encoding="utf-8",
    )
    trials = read_tune_ndjson(source)
    assert len(trials) == 1
    assert trials[0]["fitness"] == 0.4
    assert trials[0]["map50_95"] == 0.4
