# Transition Quality Model 1.0

## Purpose

Predict an engineered transition-quality target from precomputed tabular audio-analysis features. This model does not understand music, select a complete set, or render audio.

## Data and split policy

Dataset version: `fixture-1`. Feature version: `1.0`. Predefined leakage-safe splits were used without reshuffling. Labels may be automatic weak labels or imported human/consensus labels.

## Model

`HistGradientBoostingRegressor` with configuration recorded in `metrics.json`.

## Evaluation

Test metrics: `{"mae": 0.036538, "rmse": 0.043195, "pearson_correlation": 1.0}`. Deterministic baseline metrics: `{"mae": 0.463636, "rmse": 0.540835, "pearson_correlation": -0.984941}`.

## Feature importance

Permutation importance measures predictive association in this artifact; it does not establish musical causation.

## Limitations and intended use

Automatic labels are derived from deterministic planning scores, so they cannot demonstrate superiority over that same baseline. Use this model only for research on candidate-ranking features; validate on diverse, legally usable tracks with human-rated, artist/album-group-safe holdouts before any user-facing use.
