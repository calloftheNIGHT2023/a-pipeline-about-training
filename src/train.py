from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from common import (
    TARGET_COLUMN,
    add_features,
    build_preprocessor,
    evaluate_rmse,
    infer_schema,
    rounded_clipped_predictions,
    save_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--model-path", default="models/ridge_tfidf.pkl")
    parser.add_argument("--metadata-path", default="models/ridge_tfidf_metadata.json")
    parser.add_argument("--metrics-path", default="outputs/train_metrics.json")
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)

    if TARGET_COLUMN not in train_df.columns:
        msg = f"Expected target column '{TARGET_COLUMN}' in {args.train_path}."
        raise ValueError(msg)

    labeled_df = train_df.loc[train_df[TARGET_COLUMN].notna()].copy()
    if labeled_df.empty:
        msg = f"No labeled rows found in {args.train_path}."
        raise ValueError(msg)

    if args.max_train_rows is not None and args.max_train_rows < len(labeled_df):
        labeled_df = labeled_df.sample(
            n=args.max_train_rows,
            random_state=args.random_state,
        ).sort_index()

    schema = infer_schema(labeled_df)
    featured = add_features(labeled_df, schema)
    X = featured.drop(columns=[TARGET_COLUMN])
    y = featured[TARGET_COLUMN]

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(schema)),
            ("regressor", Ridge(alpha=args.alpha)),
        ]
    )

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    cv_scores = -cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=None,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)
    valid_raw = model.predict(X_valid)
    valid_rounded = rounded_clipped_predictions(valid_raw, y_train)

    holdout_rmse_raw = evaluate_rmse(y_valid, valid_raw)
    holdout_rmse_rounded = evaluate_rmse(y_valid, valid_rounded)

    model.fit(X, y)

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.model_path, "wb") as handle:
        pickle.dump(model, handle)

    save_metadata(args.metadata_path, schema, int(y.min()), int(y.max()))

    metrics = {
        "model": "Ridge",
        "alpha": args.alpha,
        "cv_folds": args.cv_folds,
        "max_train_rows": args.max_train_rows,
        "cv_rmse_mean": float(cv_scores.mean()),
        "cv_rmse_std": float(cv_scores.std()),
        "holdout_rmse_raw": holdout_rmse_raw,
        "holdout_rmse_rounded": holdout_rmse_rounded,
        "train_rows_total": int(len(train_df)),
        "train_rows_labeled": int(len(labeled_df)),
        "text_columns": schema.text_columns,
        "categorical_columns": schema.categorical_columns,
        "numeric_columns": schema.numeric_columns,
    }
    Path(args.metrics_path).write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
