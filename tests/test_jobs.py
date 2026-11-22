from datetime import datetime

import pandas as pd

from mandicast import jobs


def test_next_run_later_today_and_tomorrow():
    now = datetime(2026, 11, 17, 10, 0, tzinfo=jobs.PKT)
    assert jobs.next_run(now, "18:30") == datetime(2026, 11, 17, 18, 30, tzinfo=jobs.PKT)
    assert jobs.next_run(now, "09:00") == datetime(2026, 11, 18, 9, 0, tzinfo=jobs.PKT)
    # exactly on time -> next day, so a job never runs twice in a row
    assert jobs.next_run(now, "10:00") == datetime(2026, 11, 18, 10, 0, tzinfo=jobs.PKT)


def test_atomic_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    target = tmp_path / "sub" / "x.parquet"
    jobs.atomic_parquet(df, target)
    jobs.atomic_parquet(df.assign(a=[4, 5, 6]), target)
    assert pd.read_parquet(target)["a"].tolist() == [4, 5, 6]
    assert [p.name for p in target.parent.iterdir()] == ["x.parquet"]


def test_notify_api_unreachable(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    assert jobs.notify_api() is False


def test_refresh_survives_scrape_failure(monkeypatch):
    import mandicast.scrape as scrape

    def boom(argv):
        raise SystemExit("No price table found")

    monkeypatch.setattr(scrape, "main", boom)
    called = []
    monkeypatch.setattr(jobs, "rebuild_processed", lambda p: called.append(p))
    assert jobs.refresh({}) is False
    assert called == []
