[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/6KpniKiX)
# a-pipelinw-for-kaggle

Predict Amazon movie review star ratings for the Kaggle competition `cs-506-spring-2026-midterm`.

This repository contains a reproducible non-neural, non-boosting baseline pipeline for the Amazon movie review rating prediction task. The current pipeline is designed to be easy to rerun, extend, and document for Kaggle-style experimentation.

## Repository description

Baseline pipeline for Kaggle-style Amazon movie review star rating prediction using TF-IDF, categorical features, and ridge regression optimized for RMSE.

## Environment

- Python `3.13`
- Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Kaggle data setup

Authenticate the Kaggle CLI first.

Option 1:
- create `~/.kaggle/kaggle.json`
- set file permissions to `600`

Option 2:
- export `KAGGLE_USERNAME`
- export `KAGGLE_KEY`

Then download the competition files into `data/`:

```bash
python3 -m kaggle competitions download -c cs-506-spring-2026-midterm -p data
cd data
unzip cs-506-spring-2026-midterm.zip
cd ..
```

Expected files:
- `data/train.csv`
- `data/test.csv`
- `data/sample.csv`

Competition data note:
- `train.csv` contains both labeled and unlabeled rows
- the unlabeled rows are exactly the rows whose `Id` appears in `test.csv`
- `test.csv` is therefore only an ordering/template file for Kaggle submission

Observed dataset shape:
- `train.csv`: `139,753` rows and `9` columns
- labeled rows: `125,777`
- unlabeled rows to predict: `13,976`

Columns used in the current pipeline:
- text: `Summary`, `Text`
- categorical: `ProductId`, `UserId`
- numeric: `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Time`
- target: `Score`

## Reproducible baseline

Train the baseline model:

```bash
python3 src/train.py
```

Quick subset run for faster iteration:

```bash
python3 src/train.py --cv-folds 2 --max-train-rows 20000
```

This writes:
- `models/ridge_tfidf.pkl`
- `models/ridge_tfidf_metadata.json`
- `outputs/train_metrics.json`

Generate a Kaggle submission:

```bash
python3 src/predict.py
```

This writes:
- `outputs/submission.csv`

## Project structure

```text
src/common.py    shared schema inference, feature engineering, preprocessing
src/train.py     training and evaluation entrypoint
src/predict.py   submission generation entrypoint
data/            Kaggle data files
models/          serialized trained models
outputs/         metrics and submission files
```

## Current approach

Current baseline:
- treat the task as ordered rating prediction and optimize against RMSE
- train only on the rows in `train.csv` where `Score` is present
- predict the rows in `train.csv` where `Score` is missing, using `test.csv` only to preserve Kaggle row order
- combine all detected text columns into one review text field
- build sparse text features with word-level TF-IDF using unigrams and bigrams
- add simple engineered numeric features such as length, punctuation, uppercase ratio, and digit ratio
- fit a `Ridge` regressor and round predictions back to valid star ratings for submission

Why this is allowed:
- no deep learning
- no boosting
- uses standard linear modeling plus hand-built features

## Modeling rationale

This dataset has three distinct signal sources:
- free-form review text, which is the main source of sentiment and rating evidence
- high-cardinality review context fields such as `ProductId` and `UserId`
- metadata such as helpfulness counts and review time

The baseline separates those feature types instead of forcing everything into one representation:
- `TF-IDF` captures lexical patterns and short phrases in `Summary` and `Text`
- one-hot encoding captures recurring user and product effects
- numeric and hand-built text statistics provide additional low-dimensional signals

`Ridge` regression was chosen as the first strong baseline because it is:
- fast enough for repeated experiments on sparse features
- robust in high-dimensional text settings
- naturally aligned with the RMSE objective
- allowed under the assignment rules

## Validation strategy

- local evaluation uses `RMSE`, which matches the Kaggle competition metric
- the training script reports both cross-validation RMSE and a holdout RMSE
- a smaller subset mode is included for fast iteration before running larger experiments

Current quick-run result on a `20,000` row labeled subset:
- `2-fold CV RMSE`: about `0.8888`
- holdout RMSE on raw regression output: about `0.8696`

## Process notes

The repository is being built to satisfy the assignment requirements:
- reproducible scripts instead of notebook-only workflow
- explicit dependency list
- separate training and submission commands
- metrics saved to disk for each run
- room to add graphs, feature analysis, and experiment history

## Next steps

After the Kaggle data is available locally, the next work items are:
- compare larger full-data runs against the subset baseline
- add better helpfuness-derived and time-derived features
- test leakage-safe user and product aggregate statistics
- compare several non-boosting models
- tune hyperparameters with cross-validation
- document each Kaggle submission with a matching git commit
