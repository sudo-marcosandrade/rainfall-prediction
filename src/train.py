import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from .data import load_raw_data, prepare_dataset
from .evaluate import evaluate_classifier, plot_evaluation_figures, save_metrics
from .features import build_preprocessor
from .models import get_candidate_models


def main() -> None:
    raw_data = load_raw_data()
    x, y = prepare_dataset(raw_data)

    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    results = []
    trained_models = {}

    for name, estimator in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        results.append(evaluate_classifier(name, pipeline, x_test, y_test))
        trained_models[name] = pipeline

    metrics = save_metrics(results, REPORTS_DIR)
    best_model_name = metrics.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "rainfall_model.joblib")
    plot_evaluation_figures(best_model, x_test, y_test, FIGURES_DIR)
    _save_feature_importance(best_model, FIGURES_DIR, REPORTS_DIR)

    print(metrics.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    _run_temporal_validation(raw_data)


def _save_feature_importance(model: Pipeline, figures_dir, reports_dir) -> None:
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(estimator, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    top = importance.head(20).sort_values("importance")
    ax = top.plot.barh(x="feature", y="importance", figsize=(10, 7), legend=False)
    ax.set_title("Top 20 Most Important Features")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "feature_importance.png", dpi=160)


def _run_temporal_validation(raw_data: pd.DataFrame) -> None:
    dated_data = raw_data.copy()
    dated_data["Date"] = pd.to_datetime(dated_data["Date"], errors="coerce")
    dated_data = dated_data.dropna(subset=["Date"]).sort_values("Date")
    cutoff = int(len(dated_data) * 0.8)

    train_raw = dated_data.iloc[:cutoff]
    test_raw = dated_data.iloc[cutoff:]
    x_train, y_train = prepare_dataset(train_raw)
    x_test, y_test = prepare_dataset(test_raw)

    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    results = []
    for name, estimator in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        results.append(evaluate_classifier(name, pipeline, x_test, y_test))

    temporal_metrics = pd.DataFrame([item["metrics"] for item in results]).sort_values(
        by=["f1", "recall", "roc_auc"], ascending=False
    )
    temporal_metrics.to_csv(REPORTS_DIR / "temporal_metrics.csv", index=False)

    print("\nTemporal validation:")
    print(temporal_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
