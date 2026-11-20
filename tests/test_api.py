"""API tests against a demo bundle (fake prices + naive/ridge pyfunc models)."""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mlflow")

from fastapi.testclient import TestClient  # noqa: E402

from mandicast.api.main import create_app  # noqa: E402
from mandicast.api.settings import Settings  # noqa: E402
from mandicast.demo import build_demo  # noqa: E402

TOKEN = "test-token"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return build_demo(tmp_path_factory.mktemp("demo"), days=450)


@pytest.fixture(scope="module")
def client(bundle):
    settings = Settings(
        commodities=["onion", "potato", "tomato"], horizons=[1, 7], model_source="local",
        model_dir=bundle / "models" / "serving", data_path=bundle / "data" / "processed" / "prices.parquet",
        admin_token=TOKEN, background_load=False)
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] and body["models_loaded"] == 6 and body["bundle_source"] == "demo"


def test_models(client):
    models = client.get("/v1/models").json()
    assert len(models) == 6
    assert {(m["commodity"], m["horizon"]) for m in models} == {
        (c, h) for c in ["onion", "potato", "tomato"] for h in [1, 7]}


def test_forecast_one(client, bundle):
    r = client.get("/v1/forecast/tomato", params={"horizon": 7})
    assert r.status_code == 200
    f = r.json()["forecasts"][0]
    prices = pd.read_parquet(bundle / "data" / "processed" / "prices.parquet")
    last = prices["date"].max()
    assert f["origin_date"] == str(last.date())
    assert f["target_date"] == str((last + pd.Timedelta(days=7)).date())
    assert f["yhat"] > 0 and f["model"]["model"] == "linear"


def test_naive_forecast_equals_last_price(client, bundle):
    prices = pd.read_parquet(bundle / "data" / "processed" / "prices.parquet")
    last = prices[prices["commodity"] == "onion"].sort_values("date")["price"].iloc[-1]
    f = client.get("/v1/forecast/onion", params={"horizon": 1}).json()["forecasts"][0]
    assert f["yhat"] == pytest.approx(last)


def test_forecast_all(client):
    body = client.get("/v1/forecast").json()
    assert len(body["forecasts"]) == 6


def test_bad_inputs(client):
    assert client.get("/v1/forecast/garlic").status_code == 404
    assert client.get("/v1/forecast/tomato", params={"horizon": 3}).status_code == 422
    assert client.post("/v1/predict", json={"commodity": "tomato", "horizon": 7,
                                            "history": []}).status_code == 422


def test_prices(client):
    body = client.get("/v1/prices/potato", params={"days": 30}).json()
    assert len(body["points"]) == 30


def test_custom_predict(client):
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    history = [{"date": str(d.date()), "price": 5000.0 + i} for i, d in enumerate(dates)]
    r = client.post("/v1/predict", json={"commodity": "onion", "horizon": 1, "history": history})
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["origin_date"] == "2026-04-30"
    assert f["yhat"] == pytest.approx(5119.0)


def test_reload_needs_token(client):
    assert client.post("/admin/reload").status_code == 401
    r = client.post("/admin/reload", headers={"X-Admin-Token": TOKEN})
    assert r.status_code == 200 and r.json()["models_loaded"] == 6


def test_metrics(client):
    client.get("/v1/forecast/tomato")
    text = client.get("/metrics").text
    assert "mandicast_predictions_total" in text
    assert "mandicast_http_requests_total" in text
