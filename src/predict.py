from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common import add_features, infer_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--model-path", default="models/ridge_tfidf.pkl")
    parser.add_argument("--submission-path", default="outputs/submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # 这个比赛把无标签样本也放在 `train.csv` 里，`test.csv` 主要用于规定提交顺序。
    labeled_df = train_df.loc[train_df["Score"].notna()].copy()
    unlabeled_df = train_df.loc[train_df["Score"].isna()].copy()
    schema = infer_schema(labeled_df)

    if set(test_df[schema.id_column]) != set(unlabeled_df[schema.id_column]):
        msg = "Test IDs do not match the unlabeled rows in train.csv."
        raise ValueError(msg)

    unlabeled_ordered = (
        unlabeled_df.drop(columns=["Score"])
        .merge(test_df[[schema.id_column]], on=schema.id_column, how="right")
    )
    # 预测前要先补回训练时用过的那套人工特征，保证输入结构一致。
    featured_test = add_features(unlabeled_ordered, schema)

    with open(args.model_path, "rb") as handle:
        model = pickle.load(handle)

    raw_predictions = model.predict(featured_test)
    # 这里沿用训练阶段的后处理方式，确保提交结果落在合法评分区间内。
    rounded = np.clip(
        np.rint(raw_predictions),
        int(labeled_df["Score"].min()),
        int(labeled_df["Score"].max()),
    ).astype(int)

    submission = pd.DataFrame(
        {
            schema.id_column: unlabeled_ordered[schema.id_column],
            "Score": rounded,
        }
    )
    Path(args.submission_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    print(f"Wrote {len(submission)} predictions to {args.submission_path}")


if __name__ == "__main__":
    main()
