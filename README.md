# Freight Rate Prediction Challenge

Predict the rate of 12,000 freight loads (Nov-Dec 2025) from 48,000 labeled loads (Jan-Oct 2025).
Result: about 1.8% average error (MAPE) on held-out future months with rolling time-based validation.

## Run it

    python -m pip install -r requirements.txt
    python -m pip install lightgbm
    python predict.py      #trains the final model, writes both prediction files
    python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
    python evaluate.py     #validation experiments (~5 min) -> results/

`predict.py` writes `validation_predictions.csv` (load_id,predicted_rate, 12,000 rows) and fills the
`predicted_rate` column of `data/december_chart_inputs.csv`. The scorer then validates both and creates
`scorer_results/candidate_december.png`.

## Files

- `features.py` - loading, outlier flagging, cleaning, feature building
- `models.py` - final model: LightGBM for the lane price + a linear layer for the date effect
- `evaluate.py` - rolling time-based validation of 7 models
- `predict.py` - final fit on all cleaned data and prediction files

## Split and validation

The labeled data is Jan-Oct and the validation loads are Nov-Dec, so this is forecasting forward in time.
I never used a random split to choose anything. Five rolling folds always train on earlier months and test on the
next two (Jan-Apr then May-Jun, ..., Jan-Aug then Sep-Oct). A random split is run only to show how much it
flatters the model (1.4% vs 2.5% error for the same plain LightGBM).

## Data problems and fixes

- Corrupted rates (677 rows, 1.4%): rate is roughly 2-5x too high or low for its lane. Flagged against the local
  median rate per mile and dropped from training only.
- Weight: 292 negative values (made absolute), 1,204 at the 47,500 cap, 300 missing (equipment median plus a flag).
- market_index missing (374): filled with that day's median.
- Eight cities appear only in validation, so city names are not used as features.
- Coordinates look approximate, so they are used as rough regions.

## Model

Trees handle the nonlinear lane price but cannot extrapolate. The date effect (market_index, steady upward drift,
a quote_signal pattern, day of week) has to be extrapolated into months the labels never show, so it is a small
linear layer. Both are fitted together on log(rate). On held-out future months this gets 1.77% mean error vs
2.48% for plain LightGBM and 4.36% for a median-rate-per-mile baseline.

## Assumptions

- `december_chart_inputs.csv` has no market_index or quote_signal, so each date's median across validation.csv
  is used. Coordinates come from a city lookup built from the provided files.
- The Nov-Dec date effect is extrapolated, since there are no labels after October.