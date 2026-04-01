from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


TARGET_COLUMN = "Score"
DEFAULT_ID_CANDIDATES = ("Id", "id", "ID", "ReviewId", "review_id")
DEFAULT_TEXT_HINTS = ("summary", "text", "review")


@dataclass
class Schema:
    id_column: str
    text_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    engineered_numeric_columns: list[str]


def infer_id_column(df: pd.DataFrame) -> str:
    for column in DEFAULT_ID_CANDIDATES:
        if column in df.columns:
            return column

    candidates = [column for column in df.columns if column != TARGET_COLUMN]
    if not candidates:
        msg = "Could not infer an ID column."
        raise ValueError(msg)
    return candidates[0]


def infer_schema(df: pd.DataFrame) -> Schema:
    id_column = infer_id_column(df)

    text_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in df.columns:
        if column in {id_column, TARGET_COLUMN}:
            continue
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            name = column.lower()
            if any(hint in name for hint in DEFAULT_TEXT_HINTS):
                text_columns.append(column)
            else:
                categorical_columns.append(column)

    numeric_columns = [
        column
        for column in df.columns
        if column not in {id_column, TARGET_COLUMN}
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    engineered_numeric_columns = []
    for column in text_columns:
        engineered_numeric_columns.extend(
            [
                f"{column}__char_count",
                f"{column}__word_count",
                f"{column}__exclamation_count",
                f"{column}__question_count",
                f"{column}__uppercase_ratio",
                f"{column}__digit_ratio",
            ]
        )

    engineered_numeric_columns.extend(["merged_text__char_count", "merged_text__word_count"])
    return Schema(
        id_column=id_column,
        text_columns=text_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        engineered_numeric_columns=engineered_numeric_columns,
    )


def _uppercase_ratio(values: pd.Series) -> pd.Series:
    total = values.str.len().replace(0, np.nan)
    ratio = values.str.count(r"[A-Z]") / total
    return ratio.fillna(0.0)


def _digit_ratio(values: pd.Series) -> pd.Series:
    total = values.str.len().replace(0, np.nan)
    ratio = values.str.count(r"\d") / total
    return ratio.fillna(0.0)


def add_features(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    enriched = df.copy()

    if schema.text_columns:
        merged = (
            enriched[schema.text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        merged = pd.Series("", index=enriched.index, dtype="object")

    enriched["merged_text"] = merged
    enriched["merged_text__char_count"] = merged.str.len()
    enriched["merged_text__word_count"] = merged.str.split().str.len().fillna(0)

    for column in schema.text_columns:
        values = enriched[column].fillna("").astype(str)
        enriched[f"{column}__char_count"] = values.str.len()
        enriched[f"{column}__word_count"] = values.str.split().str.len().fillna(0)
        enriched[f"{column}__exclamation_count"] = values.str.count("!")
        enriched[f"{column}__question_count"] = values.str.count(r"\?")
        enriched[f"{column}__uppercase_ratio"] = _uppercase_ratio(values)
        enriched[f"{column}__digit_ratio"] = _digit_ratio(values)

    return enriched


def build_preprocessor(schema: Schema) -> ColumnTransformer:
    numeric_features = schema.numeric_columns + schema.engineered_numeric_columns

    text_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            )
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", MaxAbsScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, object]] = [("text", text_pipeline, "merged_text")]
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if schema.categorical_columns:
        transformers.append(("cat", categorical_pipeline, schema.categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def rounded_clipped_predictions(raw_predictions: np.ndarray, y_train: pd.Series) -> np.ndarray:
    low = int(y_train.min())
    high = int(y_train.max())
    return np.clip(np.rint(raw_predictions), low, high).astype(int)


def evaluate_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def save_metadata(path: str | Path, schema: Schema, train_score_min: int, train_score_max: int) -> None:
    payload = {
        "id_column": schema.id_column,
        "text_columns": schema.text_columns,
        "categorical_columns": schema.categorical_columns,
        "numeric_columns": schema.numeric_columns,
        "engineered_numeric_columns": schema.engineered_numeric_columns,
        "train_score_min": int(train_score_min),
        "train_score_max": int(train_score_max),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_metadata(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
