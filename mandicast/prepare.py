"""Raw AMIS rows -> one clean daily price series per commodity.

Output: data/processed/prices.parquet with columns date, commodity, price, imputed
plus reports/data_quality.json (tracked as DVC metrics).
"""
import json
import logging
import re

import numpy as np
import pandas as pd

from .config import PRICES_PATH, REPORTS_DIR, ROOT, load_params, setup_logging
from .scrape import clean_name

log = logging.getLogger("prepare")


def map_commodities(raw, patterns):
    """Attach canonical commodity names. Earlier patterns get higher priority."""
    raw = raw.copy()
    raw["commodity_raw"] = raw["commodity_raw"].map(clean_name)
    name = raw["commodity_raw"].str.lower()
    raw["commodity"] = None
    raw["priority"] = np.inf
    for commodity, pats in patterns.items():
        for prio, pat in enumerate(pats):
            hit = name.str.contains(pat, flags=re.IGNORECASE, regex=True) & (raw["priority"] > prio)
            # don't let a later commodity steal rows another one already claimed
            hit &= raw["commodity"].isna() | (raw["commodity"] == commodity)
            raw.loc[hit, "commodity"] = commodity
            raw.loc[hit, "priority"] = prio
    return raw.dropna(subset=["commodity"])


def pick_price(df):
    mid = (df["min_price"] + df["max_price"]) / 2
    price = df["fqp"].where(df["fqp"] > 0, mid)
    return price.where(price > 0)


def fix_spikes(logp, n_sigmas, min_jump, window=61):
    """Replace one-day spikes (price jumps and comes straight back, usually a data
    entry typo) with the average of the neighbours. Genuine level shifts are kept."""
    r_in = logp.diff()
    r_out = -logp.diff(-1)
    sigma = 1.4826 * (r_in - r_in.rolling(window, center=True, min_periods=10).median()).abs() \
        .rolling(window, center=True, min_periods=10).median()
    thresh = np.maximum(n_sigmas * sigma, min_jump)
    spike = ((r_in.abs() > thresh) & (r_out.abs() > thresh) & (np.sign(r_in) != np.sign(r_out))).fillna(False)
    neighbours = (logp.shift(1) + logp.shift(-1)) / 2
    return logp.where(~spike, neighbours), spike


def splice_varieties(grp, max_gap):
    """Build one daily series per commodity from the AMIS names that matched it.

    The best-priority name is used wherever it exists. A lower-priority name only
    fills stretches where the main one is missing for longer than `max_gap` days.
    AMIS stops reporting "Potato Fresh" in Lahore for months at a time and lists
    "Potato Sugar free" instead; short gaps are still interpolated from the main
    variety so the series doesn't zig-zag between the two.
    """
    names = (grp.groupby("commodity_raw")
                .agg(priority=("priority", "min"), n=("date", "size"))
                .sort_values(["priority", "n"], ascending=[True, False]).index)
    by_name = {n: g.set_index("date")["price"] for n, g in grp.groupby("commodity_raw")}
    days = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")

    price = by_name[names[0]].reindex(days)
    source = pd.Series(np.where(price.notna(), names[0], None), index=days, dtype=object)
    for name in names[1:]:
        gap = price.isna()
        run_id = (~gap).cumsum()
        run_len = gap.groupby(run_id).transform("sum")
        long_gap = gap & (run_len > max_gap)
        fill = long_gap & by_name[name].reindex(days).notna()
        price[fill] = by_name[name].reindex(days)[fill]
        source[fill] = name

    out = pd.DataFrame({"date": days, "price": price.to_numpy(), "variety": source.to_numpy()})
    out = out.dropna(subset=["price"])
    used = out["variety"].value_counts().reindex(names).dropna().astype(int)
    return out, used.to_dict()


def clean_series(df, start, max_gap, spike_sigmas, spike_min_jump):
    s = df.set_index("date")["price"].sort_index()
    s = s[s.index >= start]
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full)
    missing = s.isna()

    logp = np.log(s)
    logp = logp.interpolate(method="time", limit=max_gap, limit_area="inside")
    long_gap = logp.isna()
    logp = logp.ffill()   # anything longer than max_gap: carry the last price, but flag it

    logp, spikes = fix_spikes(logp, spike_sigmas, spike_min_jump)
    out = pd.DataFrame({
        "date": full,
        "price": np.exp(logp.to_numpy()).round(2),
        "imputed": (missing | spikes).to_numpy(),
    })
    stats = {
        "days": int(len(full)),
        "start": str(full.min().date()),
        "end": str(full.max().date()),
        "missing_days": int(missing.sum()),
        "long_gap_days": int(long_gap.sum()),
        "spikes_fixed": int(spikes.sum()),
    }
    return out, stats


def prepare(raw, params):
    p = params["prepare"]
    raw = raw[raw["city"].astype(str).str.lower() == params["data"]["city"].lower()].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for c in ["min_price", "max_price", "fqp"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=["date"])

    raw = map_commodities(raw, params["data"]["commodities"])
    raw["price"] = pick_price(raw)
    raw = raw.dropna(subset=["price"])
    raw = (raw.sort_values(["date", "commodity", "priority", "scraped_at"],
                           ascending=[True, True, True, False])
              .drop_duplicates(["date", "commodity", "commodity_raw"]))

    spliced = []
    for commodity, grp in raw.groupby("commodity"):
        out, used = splice_varieties(grp, p["max_gap_days"])
        out["commodity"] = commodity
        spliced.append(out)
        log.info("%s <- %s", commodity, ", ".join(f"'{n}' ({d} days)" for n, d in used.items()))
    raw = pd.concat(spliced, ignore_index=True)

    frames, quality = [], {}
    for commodity, grp in raw.groupby("commodity"):
        series, stats = clean_series(grp, pd.Timestamp(p["start_date"]), p["max_gap_days"],
                                     p["spike_sigmas"], p["spike_min_jump"])
        series.insert(1, "commodity", commodity)
        frames.append(series)
        quality[commodity] = stats
        log.info("%-7s %s .. %s  %4d days, %3d missing, %2d spikes",
                 commodity, stats["start"], stats["end"], stats["days"],
                 stats["missing_days"], stats["spikes_fixed"])

    wanted = set(params["data"]["commodities"])
    if not frames or set(quality) != wanted:
        raise SystemExit(f"Missing commodities in raw data: {sorted(wanted - set(quality))}")
    prices = pd.concat(frames, ignore_index=True)

    # keep the common date range so every fold sees all commodities
    lo = prices.groupby("commodity")["date"].min().max()
    hi = prices.groupby("commodity")["date"].max().min()
    prices = prices[(prices["date"] >= lo) & (prices["date"] <= hi)].reset_index(drop=True)
    quality["common_range"] = {"start": str(lo.date()), "end": str(hi.date()),
                               "days": int((hi - lo).days + 1)}
    return prices, quality


def main():
    setup_logging()
    params = load_params()
    raw = pd.read_csv(ROOT / params["data"]["raw_path"])
    prices, quality = prepare(raw, params)

    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    prices.to_parquet(PRICES_PATH, index=False)
    with open(REPORTS_DIR / "data_quality.json", "w") as f:
        json.dump(quality, f, indent=2)
    log.info("wrote %d rows -> %s", len(prices), PRICES_PATH)


if __name__ == "__main__":
    main()
