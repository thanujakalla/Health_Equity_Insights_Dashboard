from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    feature_columns: tuple[str, ...]
    target_column: str
    mae: float
    r2: float


DEFAULT_FEATURES: tuple[str, ...] = ("AGE", "INCOME", "RACE", "GENDER")
DEFAULT_TARGET = "TOTAL_CLAIM_COST"


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training data missing required columns: {missing}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )


def train_cost_predictor(
    df: pd.DataFrame,
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    target: str = DEFAULT_TARGET,
    random_state: int = 42,
) -> TrainResult:
    """
    Train a cost prediction model using a robust preprocessing + model pipeline.

    Returns a fitted sklearn Pipeline that can be directly joblib-dumped and later used
    for inference with the same feature columns.
    """
    _ensure_columns(df, list(features) + [target])

    work = df[list(features) + [target]].copy()

    # Basic cleanup: coerce numerics; leave categoricals as-is.
    for col in ("AGE", "INCOME"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # Guard against obviously invalid values
    if "AGE" in work.columns:
        work.loc[(work["AGE"] < 0) | (work["AGE"] > 110), "AGE"] = np.nan
    if "INCOME" in work.columns:
        work.loc[work["INCOME"] < 0, "INCOME"] = np.nan

    X = work[list(features)]
    y = pd.to_numeric(work[target], errors="coerce")

    # Drop rows without target (can't learn)
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    numeric_features = [c for c in ("AGE", "INCOME") if c in features]
    categorical_features = [c for c in ("RACE", "GENDER") if c in features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test.to_numpy())))
    # r2_score without importing extra: 1 - SSE/SST
    sse = float(np.sum((preds - y_test.to_numpy()) ** 2))
    sst = float(np.sum((y_test.to_numpy() - float(np.mean(y_test.to_numpy()))) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst != 0 else float("nan")

    return TrainResult(
        model=pipe,
        feature_columns=features,
        target_column=target,
        mae=mae,
        r2=r2,
    )


def save_model(model: Pipeline, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> Pipeline:
    return joblib.load(model_path)
