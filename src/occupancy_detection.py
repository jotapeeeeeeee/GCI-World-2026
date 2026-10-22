"""Reusable data and modeling functions for occupancy detection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TARGET = "Occupancy"


def load_dataset() -> pd.DataFrame:
    """Download the UCI Occupancy Detection data and return sensor readings."""
    occupancy = fetch_ucirepo(id=357)
    data = occupancy.data.features[FEATURES].copy()
    data[TARGET] = occupancy.data.targets[TARGET]
    return data.apply(pd.to_numeric, errors="coerce")


def plot_eda(
    data: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> tuple[plt.Figure, pd.Series]:
    """Plot CO2 by occupancy and sensor correlations; optionally save the figure."""
    correlation_matrix = data.corr(numeric_only=True)
    sensor_correlations = (
        correlation_matrix[TARGET]
        .drop(TARGET)
        .sort_values(key=np.abs, ascending=False)
    )

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(
        ax=axes[0],
        data=data,
        x=TARGET,
        y="CO2",
        color="steelblue",
    )
    axes[0].set_title("CO2 levels by room occupancy")
    axes[0].set_xlabel("Occupancy (0 = empty, 1 = occupied)")
    axes[0].set_ylabel("CO2")

    sns.heatmap(
        correlation_matrix,
        ax=axes[1],
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    axes[1].set_title("Correlation between sensor readings and occupancy")
    figure.tight_layout()

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path / "occupancy-eda.png", dpi=160, bbox_inches="tight")

    return figure, sensor_correlations


def evaluate_baselines(data: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    """Compare three classifiers with cross-validation on the training split."""
    features, _, target, _ = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.2,
        stratify=data[TARGET],
        random_state=42,
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    }

    results = []
    for name, model in models.items():
        scores = cross_val_score(
            model,
            features,
            target,
            cv=cv,
            scoring="accuracy",
        )
        results.append(
            {
                "Model": name,
                "Mean accuracy": scores.mean(),
                "Standard deviation": scores.std(),
            }
        )

    return pd.DataFrame(results).sort_values("Mean accuracy", ascending=False)


def tune_knn(
    data: pd.DataFrame,
    cv: int = 5,
) -> tuple[GridSearchCV, float]:
    """Tune KNN on a training split and return the search and test accuracy."""
    features = data[FEATURES]
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21]},
        cv=cv,
        scoring="accuracy",
    )
    search.fit(x_train, y_train)
    return search, search.score(x_test, y_test)
