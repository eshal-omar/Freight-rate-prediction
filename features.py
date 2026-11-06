"""Data loading, cleaning and feature engineering for the freight rate challenge."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
MAX_WEIGHT = 47_500.0  #weights are capped at this value in the raw data

LANE_FEATURES = [
    "log_distance", "distance", "equipment_code", "weight_clean",
    "weight_missing", "weight_was_negative", "weight_at_cap",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "d_lat", "d_lon", "haversine", "road_factor",
]
MARKET_FEATURES = [
    "market_index", "market_index_date", "quote_signal", "quote_signal_date", "market_index_missing",
]
CALENDAR_FEATURES = ["dow", "is_weekend"]  #no month on purpose: training only covers Jan-Oct
FEATURES = LANE_FEATURES + MARKET_FEATURES + CALENDAR_FEATURES
START_DATE = pd.Timestamp("2025-01-01")

EQUIPMENT_CODES = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}


def load_raw():
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    return train, valid


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Lookup of city -> (lat, lon). Each city has one coordinate pair in the data."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def date_level_signal(df, column, reference=None):
    """Median of a market column for every date (a de-noised, date-level version of it)."""
    source = df if reference is None else reference
    daily = source.groupby("date")[column].median()
    return df["date"].map(daily)


def build_features(df: pd.DataFrame, market_reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw rows -> model matrix. Works for train, validation and the December chart rows.

    market_reference: frame with market_index / quote_signal per date, used when the
    rows themselves do not carry them (the December chart inputs).
    """
    out = pd.DataFrame(index=df.index)

    #distance and geography
    out["distance"] = df["distance"].astype(float)
    out["log_distance"] = np.log(out["distance"])
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        out[c] = df[c].astype(float)
    out["d_lat"] = out["delivery_lat"] - out["pickup_lat"]
    out["d_lon"] = out["delivery_lon"] - out["pickup_lon"]
    out["haversine"] = haversine_miles(out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon)
    out["road_factor"] = out["distance"] / out["haversine"].clip(lower=1.0)

    #equipment
    out["equipment_code"] = df["equipment"].map(EQUIPMENT_CODES).astype(float)

    #weight: negatives are sign errors, missing get the equipment median
    w = df["weight"].astype(float)
    out["weight_missing"] = w.isna().astype(float)
    out["weight_was_negative"] = (w < 0).astype(float)
    w = w.abs().clip(upper=MAX_WEIGHT)
    out["weight_at_cap"] = (w >= MAX_WEIGHT).astype(float)
    medians = w.groupby(df["equipment"]).transform("median")
    out["weight_clean"] = w.fillna(medians).fillna(w.median())

    #market conditions
    ref = df if market_reference is None else market_reference
    out["market_index_date"] = date_level_signal(df, "market_index", ref) if "market_index" in df else df["date"].map(
        market_reference.groupby("date")["market_index"].median())
    out["quote_signal_date"] = date_level_signal(df, "quote_signal", ref) if "quote_signal" in df else df["date"].map(
        market_reference.groupby("date")["quote_signal"].median())
    mi = df["market_index"] if "market_index" in df else out["market_index_date"]
    qs = df["quote_signal"] if "quote_signal" in df else out["quote_signal_date"]
    out["market_index_missing"] = mi.isna().astype(float)
    out["market_index"] = mi.fillna(out["market_index_date"])  # same-day median fills gaps
    out["quote_signal"] = qs.fillna(out["quote_signal_date"])

    #calendar
    out["dow"] = df["date"].dt.dayofweek.astype(float)
    out["is_weekend"] = (out["dow"] >= 5).astype(float)
    out["t_days"] = (df["date"] - START_DATE).dt.days.astype(float)
    return out[FEATURES + ["t_days"]]


def flag_rate_outliers(train: pd.DataFrame, low: float = 0.6, high: float = 1.6) -> pd.Series:
    """True for rows whose rate per mile is far outside what similar loads pay.

    "Similar" = same equipment and same slice of the distance range (20 slices).
    The ratio to that local median is either within about 0.78-1.28 or beyond
    0.55 / 1.8 with nothing in between, so any cut-off from 0.6 to 1.6 flags the same rows.
    """
    rpm = train["posted_rate"] / train["distance"]
    slice_ = pd.qcut(train["distance"], 20, labels=False, duplicates="drop")
    local_median = rpm.groupby([train["equipment"], slice_]).transform("median")
    ratio = rpm / local_median
    return (ratio < low) | (ratio > high)