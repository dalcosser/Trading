import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

st.set_page_config(page_title="YF Data Diagnoser", layout="wide")
st.title("🔎 Yahoo Finance Data Diagnoser")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    st.markdown("**Fetch mode**")
    mode = st.radio("Choose", ["Period + Interval", "Start/End + Interval"], index=0)

    interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "90m", "1h", "1d", "1wk", "1mo"], index=6)

    if mode == "Period + Interval":
        period = st.selectbox("Period", ["1d","2d","5d","7d","14d","30d","60d","90d","1y","2y","5y","10y","max"], index=5)
    else:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start", value=date.today() - timedelta(days=90))
        with c2:
            end = st.date_input("End", value=date.today())

    st.markdown("---")
    st.subheader("Known Yahoo Limits")
    st.info(
        "• **1m** ≤ ~7 days\n"
        "• **2m/5m/15m/30m/90m** ≤ ~60 days\n"
        "• **1h/1d/1wk/1mo**: long ranges OK\n"
        "Tip: If `1m` returns nothing, try shorter `period` (5–7d) or higher interval."
    )

    run = st.button("Run Diagnosis", use_container_width=True)

def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame([{"rows": 0, "first": None, "last": None, "columns": []}])
    return pd.DataFrame([{
        "rows": len(df),
        "first": str(df.index[0]) if len(df) else None,
        "last":  str(df.index[-1]) if len(df) else None,
        "columns": list(df.columns)
    }])

def fetch_once(tkr: str, interval: str, period: str | None = None, start: str | None = None, end: str | None = None):
    try:
        if period:
            df = yf.download(tkr, period=period, interval=interval, progress=False, auto_adjust=False)
            meta = {"mode": "period+interval", "period": period, "interval": interval}
        else:
            df = yf.download(tkr, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
            meta = {"mode": "start-end+interval", "start": start, "end": end, "interval": interval}
        return df, meta, None
    except Exception as e:
        return None, {"interval": interval, "period": period, "start": start, "end": end}, str(e)

if not run:
    st.warning("Set inputs on the left and click **Run Diagnosis**.")
    st.stop()

# --- Single run based on chosen mode
st.subheader("Single Fetch Result")
if mode == "Period + Interval":
    df, meta, err = fetch_once(ticker, interval, period=period)
else:
    df, meta, err = fetch_once(ticker, interval, start=start.isoformat(), end=end.isoformat())

c1, c2 = st.columns([1,2])
with c1:
    st.write("**Parameters**", meta)
    if err:
        st.error(f"Exception: {err}")
with c2:
    st.write("**Summary**")
    st.dataframe(summarize_df(df), use_container_width=True)

st.markdown("**Head/Tail**")
colA, colB = st.columns(2)
with colA:
    st.caption("Head(5)")
    st.dataframe(df.head(5) if isinstance(df, pd.DataFrame) else pd.DataFrame(), use_container_width=True)
with colB:
    st.caption("Tail(5)")
    st.dataframe(df.tail(5) if isinstance(df, pd.DataFrame) else pd.DataFrame(), use_container_width=True)

st.markdown("---")

# --- Matrix test over multiple combos to see what works for the ticker
st.subheader("Matrix Test (interval × period)")
intervals = ["1m","2m","5m","15m","30m","90m","1h","1d"]
periods   = ["1d","2d","5d","7d","14d","30d","60d"]

if st.button("Run Matrix Test", use_container_width=True):
    results = []
    for i in intervals:
        for p in periods:
            d, _, err = fetch_once(ticker, i, period=p)
            rows = 0 if (d is None or d.empty) else len(d)
            first = str(d.index[0]) if rows else None
            last  = str(d.index[-1]) if rows else None
            results.append({
                "interval": i,
                "period": p,
                "rows": rows,
                "first": first,
                "last": last,
                "ok": rows > 0,
                "error": err
            })
    res_df = pd.DataFrame(results).sort_values(["interval","period"])
    st.dataframe(res_df, use_container_width=True)
    st.success("Matrix complete. Look for rows>0; that shows what Yahoo will serve for this symbol.")
