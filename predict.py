"""Train the final model on all cleaned data and write both prediction files.

Writes validation_predictions.csv and fills data/december_chart_inputs.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import DATA, build_features, city_coordinates, flag_rate_outliers, load_raw
from models import LaneDateModel

ROOT = Path(__file__).parent
FINAL_TERMS = ("market", "trend", "quote", "dow")


def main():
    train, valid = load_raw()

    # 1)drop corrupted rates, build features, fit on all of Jan-Oct
    outlier = flag_rate_outliers(train)
    X_train = build_features(train)
    keep = ~outlier
    print(f"Training rows: {keep.sum():,} of {len(train):,} ({outlier.sum()} corrupted rates removed)")
    model = LaneDateModel(terms=FINAL_TERMS).fit(X_train[keep], np.log(train.loc[keep, "posted_rate"]))

    # 2)the 12,000 validation loads, written into the template (same ids, same order)
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred = pd.Series(model.predict(build_features(valid)), index=valid["load_id"])
    assert set(template["load_id"]) == set(pred.index) and len(template) == len(valid) == 12_000
    out = template[["load_id"]].copy()
    out["predicted_rate"] = pred.reindex(out["load_id"]).to_numpy().round(2)
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"validation_predictions.csv written: {len(out):,} rows")

    # 3) the fixed December lane. The input file lacks coordinates, market_index and
    #    quote_signal, so: coordinates come from a city lookup, and the market columns
    #    are each date's median across validation.csv (which covers every December date).
    december = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    coords = city_coordinates(train, valid)
    frame = december.drop(columns="predicted_rate").copy()
    frame["pickup_lat"] = frame["pickup"].map(coords["lat"])
    frame["pickup_lon"] = frame["pickup"].map(coords["lon"])
    frame["delivery_lat"] = frame["delivery"].map(coords["lat"])
    frame["delivery_lon"] = frame["delivery"].map(coords["lon"])
    assert frame[["pickup_lat", "delivery_lat"]].notna().all().all(), "unknown city in December inputs"
    X_dec = build_features(frame, market_reference=valid)
    december["predicted_rate"] = model.predict(X_dec).round(2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("December predictions:", december["predicted_rate"].min(), "to", december["predicted_rate"].max())


if __name__ == "__main__":
    main()