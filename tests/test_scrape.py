from pathlib import Path

import pandas as pd

from mandicast.scrape import append_raw, find_page_date, parse_price_page

FIX = Path(__file__).parent / "fixtures"


def test_parse_daily_table():
    html = (FIX / "amis_daily.html").read_text()
    rows = parse_price_page(html)
    assert set(rows["commodity_raw"]) == {"Tomato", "Onion", "Potato Fresh", "Potato (Store)"}
    tomato = rows[rows["commodity_raw"] == "Tomato"].iloc[0]
    assert tomato["fqp"] == 7100 and tomato["min_price"] == 6800
    assert find_page_date(html) == pd.Timestamp("2026-11-12")


def test_parse_history_table_without_crop_column():
    rows = parse_price_page((FIX / "amis_history.html").read_text())
    assert len(rows) == 3
    assert list(rows["date"]) == list(pd.to_datetime(["2026-11-01", "2026-11-02", "2026-11-03"]))
    assert rows["fqp"].tolist() == [6750, 6900, 6800]


def test_append_raw_dedupes(tmp_path):
    out = tmp_path / "raw.csv"
    row = {"date": "2026-11-12", "city": "Lahore", "commodity_raw": "Tomato", "min_price": 1,
           "max_price": 2, "fqp": 1.5, "source": "amis_daily"}
    append_raw(pd.DataFrame([row | {"scraped_at": "2026-11-12T09:00:00"}]), out)
    append_raw(pd.DataFrame([row | {"fqp": 1.8, "scraped_at": "2026-11-12T18:00:00"}]), out)
    raw = pd.read_csv(out)
    assert len(raw) == 1 and raw["fqp"].iloc[0] == 1.8


def test_parse_real_amis_layout():
    # same structure as ViewPrices.aspx: td header with "Dated:", numbered crop
    # names, section rows, and everything nested in layout tables
    html = (FIX / "amis_viewprices.html").read_text()
    rows = parse_price_page(html)
    assert set(rows["commodity_raw"]) == {"Rice Basmati Super (New)", "Potato Fresh", "Potato Store",
                                           "Onion", "Tomato"}
    assert (rows["date"] == pd.Timestamp("2026-11-13")).all()
    tomato = rows[rows["commodity_raw"] == "Tomato"].iloc[0]
    assert (tomato["min_price"], tomato["max_price"], tomato["fqp"]) == (5200, 6000, 5600)
    # the date box says 11/13/2026 (US order) but the table header wins
    assert find_page_date(html) == pd.Timestamp("2026-11-13")


def test_history_posts_date_box(monkeypatch, tmp_path):
    import mandicast.scrape as sc

    page = (FIX / "amis_viewprices.html").read_text()
    posted = []

    class Resp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    class Session:
        def post(self, url, data, headers, timeout):
            posted.append(data)
            d = pd.Timestamp(data[sc.DATE_FIELD])          # MM/DD/YYYY
            return Resp(page.replace("Dated:13-11-2026", f"Dated:{d:%d-%m-%Y}"))

    monkeypatch.setattr(sc.requests, "Session", Session)
    monkeypatch.setattr(sc, "fetch", lambda url: page)
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)
    out = tmp_path / "raw.csv"
    sc.main(["--out", str(out), "history", "--start", "2026-11-01", "--end", "2026-11-03"])

    assert [p[sc.DATE_FIELD] for p in posted] == ["11/01/2026", "11/02/2026", "11/03/2026"]
    assert posted[0]["__VIEWSTATE"] == "abc123"
    raw = pd.read_csv(out)
    assert sorted(raw["date"].unique()) == ["2026-11-01", "2026-11-02", "2026-11-03"]

    # running again skips days that are already saved
    posted.clear()
    sc.main(["--out", str(out), "history", "--start", "2026-11-01", "--end", "2026-11-03"])
    assert posted == []
