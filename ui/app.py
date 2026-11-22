"""MandiCast dashboard (Streamlit). Talks only to the inference API.

    API_URL=http://localhost:8000 streamlit run ui/app.py
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", API_URL)   # what we show users (behind nginx: /api)
COMMODITIES = ["tomato", "onion", "potato"]

st.set_page_config(page_title="MandiCast", page_icon="🥔", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def get(path, **params):
    r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def ready():
    try:
        r = requests.get(f"{API_URL}/ready", timeout=10)
        return r.json()
    except Exception as e:
        return {"ready": False, "error": str(e)}


def rs(x):
    return "–" if x is None else f"Rs {x:,.0f}"


st.title("MandiCast")
st.caption("Wholesale price forecasts for the Lahore mandi (AMIS.pk data, Rs per 100 kg).")

status = ready()
if not status.get("ready"):
    st.warning("The API is starting up or has no models yet. "
               f"Details: {status.get('errors') or status.get('error') or status}")
    st.stop()
if status.get("bundle_source") == "demo":
    st.info("Demo mode: these forecasts come from synthetic prices, not real AMIS data.")

forecasts = get("/v1/forecast")
fc = pd.DataFrame([{**f, "model_name": f["model"]["model"], "version": f["model"]["version"]}
                   for f in forecasts["forecasts"]])

st.subheader(f"Latest prices and forecasts (data up to {forecasts['data_last_date']})")
cols = st.columns(len(COMMODITIES))
for col, c in zip(cols, COMMODITIES):
    sub = fc[fc["commodity"] == c].set_index("horizon")
    if sub.empty:
        continue
    last = get(f"/v1/prices/{c}", days=1)["points"][-1]["price"]
    with col:
        st.markdown(f"### {c.title()}")
        st.metric("Today", rs(last))
        for h in sorted(sub.index):
            y = sub.loc[h, "yhat"]
            st.metric(f"In {h} day{'s' if h > 1 else ''} ({sub.loc[h, 'target_date']})", rs(y),
                      delta=f"{(y / last - 1) * 100:+.1f}%" if y else None, delta_color="inverse")

st.divider()
left, right = st.columns([3, 1])
with right:
    commodity = st.selectbox("Commodity", COMMODITIES)
    days = st.slider("History shown (days)", 30, 365, 120, step=15)
with left:
    hist = pd.DataFrame(get(f"/v1/prices/{commodity}", days=days)["points"])
    hist["date"] = pd.to_datetime(hist["date"])
    sub = fc[fc["commodity"] == commodity].copy()
    sub["date"] = pd.to_datetime(sub["target_date"])
    # draw the forecast as a line from today's price to each horizon
    anchor = pd.DataFrame({"date": [hist["date"].max()], "yhat": [hist["price"].iloc[-1]]})
    path = pd.concat([anchor, sub[["date", "yhat"]]]).sort_values("date")

    base = alt.Chart(hist).mark_line(color="#444").encode(
        x=alt.X("date:T", title=None), y=alt.Y("price:Q", title="Rs / 100 kg", scale=alt.Scale(zero=False)),
        tooltip=["date:T", alt.Tooltip("price:Q", format=",.0f")])
    fline = alt.Chart(path).mark_line(color="#d62728", strokeDash=[4, 3]).encode(x="date:T", y="yhat:Q")
    fpts = alt.Chart(sub).mark_point(color="#d62728", filled=True, size=70).encode(
        x="date:T", y="yhat:Q",
        tooltip=["horizon", "target_date", alt.Tooltip("yhat:Q", format=",.0f"), "model_name"])
    layers = [base, fline, fpts]
    band = sub.dropna(subset=["q10", "q90"])
    if not band.empty:
        layers.append(alt.Chart(band).mark_rule(color="#d62728", strokeWidth=3, opacity=0.4)
                      .encode(x="date:T", y="q10:Q", y2="q90:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=360), use_container_width=True)
    st.caption("Dashed red: forecast; vertical bar: 80% interval (quantile models only).")

tab_models, tab_custom, tab_api = st.tabs(["Models in production", "Forecast your own series", "API"])

with tab_models:
    models = pd.DataFrame(get("/v1/models"))
    shown = ["name", "version", "alias", "model", "family", "cv_mase", "trained_until", "source"]
    st.dataframe(models[shown], hide_index=True, use_container_width=True)
    st.caption("cv_mase: walk-forward MASE from the benchmark; below 1 beats the naive forecast "
               "on the training period's scale.")

with tab_custom:
    st.write("Upload a CSV with `date` and `price` columns (daily, Rs/100kg, at least 30 days).")
    up = st.file_uploader("CSV", type="csv")
    cc1, cc2 = st.columns(2)
    c_commodity = cc1.selectbox("Commodity ", COMMODITIES, key="custom_c")
    c_h = cc2.selectbox("Horizon (days)", [1, 7], index=1)
    if up is not None:
        df = pd.read_csv(up)
        df.columns = [c.lower().strip() for c in df.columns]
        if not {"date", "price"} <= set(df.columns):
            st.error("CSV needs `date` and `price` columns.")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date", "price"])
            payload = {"commodity": c_commodity, "horizon": c_h,
                       "history": df[["date", "price"]].to_dict("records")}
            r = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=120)
            if r.ok:
                f = r.json()
                st.success(f"{c_commodity.title()} on {f['target_date']}: **{rs(f['yhat'])}** "
                           f"(80% interval {rs(f['q10'])} – {rs(f['q90'])}, model {f['model']['model']})")
            else:
                st.error(f"API error {r.status_code}: {r.text[:300]}")

with tab_api:
    public = PUBLIC_API_URL
    if public.startswith("/"):              # behind nginx: build the full URL from the request
        host = st.context.headers.get("host", "localhost")
        public = f"http://{host}{public}"
    st.write(f"Interactive docs: [{public}/docs]({public}/docs)")
    st.code(f"curl '{public}/v1/forecast/tomato?horizon=7'\n"
            f"curl '{public}/v1/forecast'\n"
            f"curl '{public}/v1/models'", language="bash")
