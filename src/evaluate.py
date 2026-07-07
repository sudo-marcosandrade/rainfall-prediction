import json
import os
from pathlib import Path

from .config import CACHE_DIR

matplotlib_cache = CACHE_DIR / "matplotlib"
temp_cache = CACHE_DIR / "tmp"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
temp_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("TMP", str(temp_cache))
os.environ.setdefault("TEMP", str(temp_cache))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(name, model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    y_score = _prediction_scores(model, x_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score) if y_score is not None else None,
    }
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    return {"metrics": metrics, "report": report, "confusion_matrix": matrix.tolist()}


def save_metrics(results: list[dict], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([item["metrics"] for item in results]).sort_values(
        by=["f1", "recall", "roc_auc"], ascending=False
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    return metrics


def plot_evaluation_figures(model, x_test, y_test, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(x_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No rain", "Rain"],
        cmap="Blues",
        values_format="d",
    )
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    y_score = _prediction_scores(model, x_test)
    if y_score is None:
        return

    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title("ROC Curve - Best Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_score)
    plt.title("Precision-Recall Curve - Best Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curve.png", dpi=160)
    plt.close()


def _prediction_scores(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return None
