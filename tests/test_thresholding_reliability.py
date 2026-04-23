import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from system import BugPredictionSystem  # noqa: E402


DATA_PATH = str(ROOT / "data" / "processed" / "cleaned_dataset.csv")


@pytest.fixture(scope="session")
def pipeline_results():
    system = BugPredictionSystem()
    system.run_complete_pipeline(DATA_PATH)
    return system


def test_threshold_selected_on_validation(pipeline_results):
    for metrics in pipeline_results.evaluation_results.values():
        assert metrics.get("threshold_selected_on") == "validation"


def test_threshold_range_and_presence(pipeline_results):
    for model_name, metrics in pipeline_results.evaluation_results.items():
        threshold = metrics.get("decision_threshold")
        assert threshold is not None, f"Missing threshold for {model_name}"
        assert 0.0 <= float(threshold) <= 1.0, f"Threshold out of range for {model_name}: {threshold}"


def test_threshold_metadata_fields_present(pipeline_results):
    for model_name, metrics in pipeline_results.evaluation_results.items():
        assert "default_metrics" in metrics, f"default_metrics missing for {model_name}"
        assert "threshold_gain" in metrics, f"threshold_gain missing for {model_name}"

        gain = metrics["threshold_gain"]
        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            assert metric_name in gain, f"{metric_name} gain missing for {model_name}"
            assert isinstance(gain[metric_name], (int, float)), f"Invalid gain type for {model_name}:{metric_name}"
            assert not math.isnan(gain[metric_name]), f"NaN gain for {model_name}:{metric_name}"


def test_thresholds_are_deterministic():
    system_a = BugPredictionSystem()
    system_a.run_complete_pipeline(DATA_PATH)
    thresholds_a = {
        model_name: metrics.get("decision_threshold")
        for model_name, metrics in system_a.evaluation_results.items()
    }

    system_b = BugPredictionSystem()
    system_b.run_complete_pipeline(DATA_PATH)
    thresholds_b = {
        model_name: metrics.get("decision_threshold")
        for model_name, metrics in system_b.evaluation_results.items()
    }

    assert thresholds_a == thresholds_b
