# Transition Quality Model Card

## Purpose

This model predicts an engineered transition-quality target from precomputed tabular representations. It is a learned regression baseline, not a system that understands music or a renderer that creates audio transitions.

## Training data and splits

Use only versioned Phase 5 transition datasets. Preserve the dataset’s predefined group-safe train, validation, and test splits; do not reshuffle candidate rows. Record dataset, feature, analysis, and model versions for every run.

## Model and features

The initial model is `HistGradientBoostingRegressor`. Inputs are numeric tempo, harmony, rhythm, energy, structure, vocal, spectral, and candidate-transition features prepared by the shared inference/training feature schema.

## Evaluation

Report MAE, RMSE, and Pearson correlation for both the learned regressor and the deterministic plan-score baseline on held-out data. Feature permutation importance describes predictive association only; it does not establish musical causation.

## Limitations

Automatic weak labels are derived from deterministic plan scores. The same deterministic plan score consequently has zero error on automatic labels, so no learned model can demonstrate an improvement on that target. A meaningful result requires diverse, legally usable audio, artist/album/source-group-safe held-out tracks, and human or consensus quality labels.

## Intended and non-intended use

Use this artifact to research whether engineered transition representations support quality prediction. Do not represent its scores as human judgments, claim it generalizes without a diverse held-out evaluation, use it to make a final DJ set without review, or use it to render audio.
