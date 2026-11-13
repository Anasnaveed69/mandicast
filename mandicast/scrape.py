"""Collect Lahore wholesale prices from AMIS.pk into data/raw.

Three ways in, all writing the same raw schema:

    python -m mandicast.scrape daily                 # today's Lahore table
    python -m mandicast.scrape history --start 2025-01-01   # backfill, one day per request
    python -m mandicast.scrape import-html pages/*.html --commodity Tomato
    python -m mandicast.scrape import-csv old_prices.csv

`daily` is what the scheduled job runs. `history` uses the "select old prices"
date box on the same page to backfill past days. The import commands are for
pages saved by hand or CSVs a teammate already has. Prices are Rs per 100 kg.
"""
import argparse
import logging
import re
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from .config import ROOT, load_params, setup_logging

log = logging.getLogger("scrape")

BASE_URL = "http://www.amis.pk"
DAILY_URL = BASE_URL + "/ViewPrices.aspx?searchType=1&commodityId={city_id}"
CITY_IDS = {"lahore": 1, "faisalabad": 2, "okara": 4, "rawalpindi": 6, "multan": 7}
HEADERS = {
    "User-Agent": "MandiCast/0.1 (FAST-NU Lahore MLOps course project)",
    "Accept-Language": "en-US,en;q=0.8",
}
PKT = ZoneInfo("Asia/Karachi")

# ViewPrices.aspx form fields (ASP.NET WebForms postback)
DATE_FIELD = "ctl00$cphPage$DateTextBox"      # expects MM/DD/YYYY
SUBMIT_FIELD = "ctl00$cphPage$ReminderButton"

RAW_COLUMNS = ["date", "city", "commodity_raw", "min_price", "max_price", "fqp",
               "source", "scraped_at"]


def fetch(url, retries=3, backoff=5.0, timeout=30):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            log.warning("attempt %d/%d failed for %s: %s", attempt, retries, url, e)
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)


def _to_number(s):
    s = s.astype(str).str.replace(",", "", regex=False)
    s = s.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce")


def _flatten(col):
    if isinstance(col, tuple):
        col = " ".join(str(c) for c in col if not str(c).startswith("Unnamed"))
    return re.sub(r"\s+", " ", str(col)).strip().lower()


def _pick(cols, *keys):
    for key in keys:
        for c in cols:
            if key in c:
                return c
    return None


def clean_name(name):
    """'48&nbspTomato' / '48\xa0Tomato' -> 'Tomato' (AMIS writes &nbsp without the ';')."""
    name = re.sub(r"&nbsp;?", " ", str(name)).replace("\xa0", " ")
    return re.sub(r"^\s*\d+\s+", "", name).strip()


def _normalise_table(df):
    """Map an arbitrary AMIS table onto commodity/min/max/fqp(/date) or return None."""
    df = df.copy()
    df.columns = [_flatten(c) for c in df.columns]

    # some GridViews come back without a header row; try the first row instead
    if all(re.fullmatch(r"\d+", c) for c in df.columns) and len(df) > 1:
        df.columns = [_flatten(c) for c in df.iloc[0]]
        df = df.iloc[1:]

    cols = list(df.columns)
    # AMIS city tables put the date in the first header cell ("Dated:13-11-2026")
    # and the crop names underneath it
    dated = next((c for c in cols if re.match(r"dated\s*:", c)), None)
    table_date = None
    if dated:
        m = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", dated)
        if m:
            table_date = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        cols = [c for c in cols if c != dated]

    name_col = dated or _pick(cols, "crop", "commodity", "item", "name")
    min_col = _pick(cols, "min")
    max_col = _pick(cols, "max")
    fqp_col = _pick(cols, "fqp", "fair", "avg", "average")
    date_col = _pick(cols, "date")
    has_price = fqp_col or (min_col and max_col)
    # history tables are one crop over many dates and may not name the crop per row
    if not has_price or (name_col is None and date_col is None):
        return None
    if dated:
        df = df.rename(columns={dated: "dated"})
        name_col = "dated"

    out = pd.DataFrame(index=df.index)
    if name_col:
        # "12 Tomato" -> "Tomato"
        out["commodity_raw"] = df[name_col].astype(str).map(clean_name)
    else:
        out["commodity_raw"] = ""
    out["min_price"] = _to_number(df[min_col]) if min_col else float("nan")
    out["max_price"] = _to_number(df[max_col]) if max_col else float("nan")
    out["fqp"] = _to_number(df[fqp_col]) if fqp_col else float("nan")
    if table_date is not None:
        out["date"] = table_date
    elif date_col:
        out["date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    if name_col:
        out = out[out["commodity_raw"].str.len() > 0]
    out = out[out[["min_price", "max_price", "fqp"]].notna().any(axis=1)]
    if not len(out):
        return None
    out.attrs["amis_city_table"] = bool(dated)
    return out


def find_page_date(html):
    text = BeautifulSoup(html, "lxml").get_text(" ")
    m = re.search(r"Dated\s*:\s*(\d{1,2})-(\d{1,2})-(\d{4})", text)
    if m:
        return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    patterns = [
        r"\b\d{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/ ,]+\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return pd.Timestamp(dateparser.parse(m.group(0), dayfirst=True).date())
            except (ValueError, OverflowError):
                continue
    return None


def parse_price_page(html):
    """Return every price row found in an AMIS page (may include a date column)."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return pd.DataFrame()
    found = [t for t in (_normalise_table(t) for t in tables) if t is not None]
    # the page is built from nested layout tables; if the real price table is
    # there, ignore whatever the outer ones parse into
    city_tables = [t for t in found if t.attrs.get("amis_city_table")]
    found = city_tables or found
    if not found:
        return pd.DataFrame()
    # nested layout tables repeat the same rows, keep the smallest complete parse
    found.sort(key=len)
    rows = pd.concat(found, ignore_index=True)
    key = [c for c in ["date", "commodity_raw"] if c in rows.columns]
    return rows.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def append_raw(new, raw_path):
    raw_path = Path(raw_path)
    new = new.reindex(columns=RAW_COLUMNS)
    new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
    if raw_path.exists():
        old = pd.read_csv(raw_path)
        new = pd.concat([old, new], ignore_index=True)
    before = len(new)
    new = (new.sort_values("scraped_at")
              .drop_duplicates(subset=["date", "city", "commodity_raw", "source"], keep="last")
              .sort_values(["date", "commodity_raw"]))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(raw_path, index=False)
    log.info("raw file now has %d rows (%d duplicates dropped) -> %s",
             len(new), before - len(new), raw_path)
    return new


def _hidden_fields(html):
    form = BeautifulSoup(html, "lxml").find("form")
    if form is None:
        raise RuntimeError("no form on the page")
    return {i["name"]: i.get("value", "") for i in form.find_all("input", type="hidden")
            if i.get("name")}


def fetch_day(session, url, fields, day, timeout=90):
    """POST the date box. Leaving the commodity checkboxes empty returns all of them."""
    data = dict(fields)
    data.update({"__EVENTTARGET": "", "__EVENTARGUMENT": "",
                 DATE_FIELD: day.strftime("%m/%d/%Y"), SUBMIT_FIELD: "Show prices"})
    resp = session.post(url, data=data, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _stamp(df, city, source):
    df = df.copy()
    df["city"] = city
    df["source"] = source
    df["scraped_at"] = datetime.now(PKT).isoformat(timespec="seconds")
    return df


def cmd_daily(args, params):
    city = params["data"]["city"]
    url = DAILY_URL.format(city_id=CITY_IDS[city.lower()])
    html = fetch(url)
    if args.save_html:
        Path(args.save_html).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(PKT).strftime("%Y%m%d_%H%M")
        (Path(args.save_html) / f"amis_{city.lower()}_{stamp}.html").write_text(html, encoding="utf-8")

    rows = parse_price_page(html)
    if rows.empty:
        raise SystemExit("No price table found on the page. Re-run with --save-html and check the layout.")
    page_date = find_page_date(html)
    if page_date is None:
        page_date = pd.Timestamp(datetime.now(PKT).date())
        log.warning("could not read the date from the page, using today (%s)", page_date.date())
    rows["date"] = page_date
    log.info("parsed %d commodities for %s on %s", len(rows), city, page_date.date())
    append_raw(_stamp(rows, city, "amis_daily"), args.out or ROOT / params["data"]["raw_path"])


def cmd_history(args, params):
    city = params["data"]["city"]
    url = DAILY_URL.format(city_id=CITY_IDS[city.lower()])
    raw_path = Path(args.out or ROOT / params["data"]["raw_path"])
    yesterday = pd.Timestamp(datetime.now(PKT).date()) - pd.Timedelta(days=1)
    end = pd.Timestamp(args.end) if args.end else yesterday
    days = list(pd.date_range(args.start, end, freq="D"))

    if raw_path.exists():
        have = set(pd.read_csv(raw_path, usecols=["date"])["date"].astype(str))
        days = [d for d in days if d.strftime("%Y-%m-%d") not in have]
    log.info("%d days to fetch (%s .. %s), already-saved days are skipped",
             len(days), args.start, end.date())
    if not days:
        return

    session = requests.Session()
    fields = _hidden_fields(fetch(url))
    buffer, empty = [], 0

    def flush():
        if buffer:
            append_raw(_stamp(pd.concat(buffer, ignore_index=True), city, "amis_history"), raw_path)
            buffer.clear()

    try:
        for i, day in enumerate(days, 1):
            for attempt in range(1, 4):
                try:
                    html = fetch_day(session, url, fields, day)
                    break
                except requests.RequestException as e:
                    log.warning("%s attempt %d failed: %s", day.date(), attempt, e)
                    if attempt == 3:
                        raise
                    time.sleep(10 * attempt)
                    fields = _hidden_fields(fetch(url))   # viewstate may have expired
            fields = _hidden_fields(html)

            rows = parse_price_page(html)
            got = rows["date"].iloc[0] if len(rows) and "date" in rows else None
            if rows.empty or got != day:
                # holidays / days the mandi didn't report
                empty += 1
                log.info("[%d/%d] %s: no prices", i, len(days), day.date())
            else:
                buffer.append(rows)
                log.info("[%d/%d] %s: %d rows", i, len(days), day.date(), len(rows))
            if len(buffer) >= 20:
                flush()
            time.sleep(args.delay)
    finally:
        flush()   # keep what we have if it stops half way (Ctrl+C, network)
    log.info("done, %d days had no prices", empty)


def cmd_import_html(args, params):
    city = args.city or params["data"]["city"]
    frames = []
    for path in args.files:
        html = Path(path).read_text(encoding="utf-8", errors="ignore")
        rows = parse_price_page(html)
        if rows.empty:
            log.warning("%s: no price table found, skipped", path)
            continue
        if "date" not in rows.columns or rows["date"].isna().all():
            d = find_page_date(html)
            if d is None:
                log.warning("%s: rows have no date and the page has none either, skipped", path)
                continue
            rows["date"] = d
        if args.commodity:
            # history pages list one crop over many dates, the crop name is
            # often only in the page heading
            rows["commodity_raw"] = args.commodity
        elif (rows["commodity_raw"] == "").any():
            raise SystemExit(f"{path}: table has no crop column, pass --commodity")
        frames.append(rows.dropna(subset=["date"]))
        log.info("%s: %d rows", path, len(frames[-1]))
    if not frames:
        raise SystemExit("Nothing imported.")
    append_raw(_stamp(pd.concat(frames, ignore_index=True), city, "amis_html"),
               args.out or ROOT / params["data"]["raw_path"])


def cmd_import_csv(args, params):
    df = pd.read_csv(args.file)
    df.columns = [_flatten(c) for c in df.columns]
    rename = {}
    for target, keys in [("date", ["date"]),
                         ("commodity_raw", ["crop", "commodity", "item"]),
                         ("min_price", ["min"]),
                         ("max_price", ["max"]),
                         ("fqp", ["fqp", "fair", "avg", "average", "price"])]:
        col = _pick([c for c in df.columns if c not in rename], *keys)
        if col is not None:
            rename[col] = target
    df = df.rename(columns=rename)
    missing = {"date", "commodity_raw"} - set(df.columns)
    if missing or not ({"fqp", "min_price"} & set(df.columns)):
        raise SystemExit(f"CSV needs date, commodity and a price column; found {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], dayfirst=args.dayfirst, errors="coerce")
    for c in ["min_price", "max_price", "fqp"]:
        df[c] = _to_number(df[c]) if c in df.columns else float("nan")
    city = args.city or params["data"]["city"]
    if "city" in df.columns:
        df = df[df["city"].astype(str).str.lower() == city.lower()]
    df = df.dropna(subset=["date"])
    log.info("importing %d rows from %s", len(df), args.file)
    append_raw(_stamp(df, city, "csv_import"), args.out or ROOT / params["data"]["raw_path"])


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None, help="raw csv to append to (default from params)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daily", help="scrape today's prices")
    d.add_argument("--save-html", default=None, help="also keep the fetched page in this folder")

    hist = sub.add_parser("history", help="backfill past days through the date box")
    hist.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    hist.add_argument("--end", default=None, help="last day (default: yesterday)")
    hist.add_argument("--delay", type=float, default=1.5, help="seconds between requests")

    h = sub.add_parser("import-html", help="parse saved AMIS pages")
    h.add_argument("files", nargs="+")
    h.add_argument("--commodity", default=None, help="crop name if the table has none per row")
    h.add_argument("--city", default=None)

    c = sub.add_parser("import-csv", help="import an existing csv")
    c.add_argument("file")
    c.add_argument("--city", default=None)
    c.add_argument("--no-dayfirst", dest="dayfirst", action="store_false")

    args = ap.parse_args(argv)
    params = load_params(args.params)
    commands = {"daily": cmd_daily, "history": cmd_history,
                "import-html": cmd_import_html, "import-csv": cmd_import_csv}
    commands[args.cmd](args, params)


if __name__ == "__main__":
    main()
