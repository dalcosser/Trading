import streamlit as st
import pandas as pd
from datetime import date as _dt_date, timedelta as _dt_timedelta

from src.ta_toolkit.data.downloader import get_ohlcv, get_options_chain
from src.ta_toolkit.strategies.rsi_reversal import rsi_reversal
from src.ta_toolkit.strategies.bbands_breakout import bbands_breakout
from src.ta_toolkit.backtest.engine import vectorized_backtest

try:
    from src.ta_toolkit.charts.price_chart import plot_candles_with_signal as plot_func
except ImportError:
    from src.ta_toolkit.charts.price_chart import plot_chart as plot_func


# ---------------------------------------------------------
st.set_page_config(page_title="Codex TA Toolkit", layout="wide")
st.markdown("# Codex TA Toolkit — Technical Analysis, Backtesting & Options")

# ---------------- Sidebar Controls -----------------------
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    # --- Interval & Period controls ---
    st.markdown("### Timeframe")
    interval = st.selectbox(
        "Interval",
        ["1d", "1h", "30m", "15m", "5m", "1m"],
        index=0,
    )

    intraday = interval in {"1h", "30m", "15m", "5m", "1m"}
    if intraday:
        period = st.selectbox(
            "Period (for intraday)",
            ["1d", "5d", "7d", "14d", "30d", "60d", "90d"],
            index=2 if interval == "1m" else 3,
        )
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start (display only)", value=_dt_date.today() - _dt_timedelta(days=7))
        with c2:
            end = st.date_input("End (display only)", value=_dt_date.today())
    else:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start", value=_dt_date.today() - _dt_timedelta(days=365 * 2))
        with c2:
            end = st.date_input("End", value=_dt_date.today())
        period = None

    # Overlays
    st.markdown("### Indicator Overlays")
    show_bbands = st.checkbox("Bollinger Bands", value=True)
    show_vwap = st.checkbox("VWAP", value=True)
    show_macd = st.checkbox("MACD (overlay)", value=False)

    with st.expander("Moving Averages", expanded=True):
        st.caption("Check which SMA/EMA to overlay")
        ma_periods = [10, 21, 50, 100, 150, 200]
        cols = st.columns(2)
        with cols[0]:
            st.write("**SMA**")
            sma_selected = [p for p in ma_periods if st.checkbox(f"SMA {p}", value=(p in [21, 50, 200]), key=f"sma_{p}")]
        with cols[1]:
            st.write("**EMA**")
            ema_selected = [p for p in ma_periods if st.checkbox(f"EMA {p}", value=(p in [21, 50]), key=f"ema_{p}")]

    # Lower panels via dropdown
    st.markdown("### Lower Panels")
    lower_options = ["RSI", "Stochastic %K/%D"]
    lower_selected = st.multiselect("Add panels", options=lower_options, default=["RSI"])
    show_rsi = "RSI" in lower_selected
    rsi_len = st.number_input("RSI Length", value=14, min_value=2, max_value=200, step=1, disabled=not show_rsi)
    show_sto = "Stochastic %K/%D" in lower_selected
    sto_k = st.number_input("%K Length", value=14, min_value=2, max_value=200, step=1, disabled=not show_sto)
    sto_d = st.number_input("%D Length", value=3, min_value=1, max_value=50, step=1, disabled=not show_sto)
    sto_smooth = st.number_input("%K Smoothing", value=3, min_value=1, max_value=50, step=1, disabled=not show_sto)

    # Backtesting (bottom)
    st.markdown("### Backtesting")
    strategy_name = st.selectbox("Strategy", ["RSI Reversal", "Bollinger Breakout"])
    fee_bps = st.number_input("Fees (bps per turn)", value=2.0, step=1.0)

    # Appearance (bottom)
    st.markdown("### Appearance")
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
    template = "plotly_dark" if theme == "Dark" else "plotly_white"


# ---------------- Utility Functions -----------------------
def _fmt(v, suffix=""):
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}{suffix}"
    except Exception:
        return str(v)


def _strategy(df: pd.DataFrame, name: str):
    if name == "RSI Reversal":
        return rsi_reversal(df)
    elif name == "Bollinger Breakout":
        return bbands_breakout(df)
    else:
        return pd.Series(0, index=df.index, name="signal")


# ---------------- Tabs -----------------------
tab1, tab2 = st.tabs(["Backtest", "Options"])


# ---------------- Tab 1: Backtest --------------------------
with tab1:
    try:
        # Fetch OHLCV data
        df = get_ohlcv(
            ticker,
            start.isoformat() if not intraday else None,
            end.isoformat() if not intraday else None,
            interval=interval,
            period=period,
        )
        st.caption(f"Data rows: {len(df)} | Columns: {list(df.columns)} | Interval: {interval}")

        # Generate strategy signal
        signal = _strategy(df, strategy_name)

        # Run backtest
        result = vectorized_backtest(df, signal, price_col="Close", fee_bps=float(fee_bps))

        # Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Return", _fmt(result.stats.get("Total_Return_%"), "%"))
        m2.metric("CAGR", _fmt(result.stats.get("CAGR_%"), "%"))
        m3.metric("Sharpe (ann.)", _fmt(result.stats.get("Sharpe")))
        m4.metric("Max Drawdown", _fmt(result.stats.get("Max_Drawdown_%"), "%"))
        m5.metric("Win Rate", _fmt(result.stats.get("Win_Rate_%"), "%"))
        m6.metric("# Trades", f"{int(result.stats.get('Num_Trades', 0))}")

        # Build overlays and indicators
        overlays = {
            "bbands": show_bbands,
            "vwap": show_vwap,
            "macd": show_macd,
            "sma_periods": sma_selected,
            "ema_periods": ema_selected,
        }
        indicators = {
            "rsi": {"show": show_rsi, "length": int(rsi_len) if show_rsi else 14},
            "stoch": {"show": show_sto, "k_len": int(sto_k) if show_sto else 14, "d_len": int(sto_d) if show_sto else 3, "smooth_k": int(sto_smooth) if show_sto else 3},
        }

        # Plot chart with indicators
        try:
            fig = plot_func(df, signal, ticker, overlays, indicators, template=template)
        except TypeError:
            fig = plot_func(df, signal, ticker, overlays)
        st.plotly_chart(fig, use_container_width=True)

        # Equity Curve
        st.subheader("Equity Curve")
        eq = result.equity_curve.copy()
        eq.name = "Equity Curve"
        st.line_chart(eq)

        # Trade Log
        st.subheader("Trade Log")
        if result.trades.empty:
            st.info("No trades for the selected period.")
        else:
            st.dataframe(result.trades, use_container_width=True)

    except Exception as e:
        st.error(f"Backtest error: {e!s}")
        st.exception(e)


# ---------------- Tab 2: Options --------------------------
with tab2:
    st.markdown("### Nearest Options Chain (sample)")
    try:
        calls, puts, expirations = get_options_chain(ticker, nearest=True)
        if len(expirations) == 0:
            st.warning("No options data available for this ticker.")
        else:
            exp = calls["expiration"].iloc[0] if not calls.empty else (
                puts["expiration"].iloc[0] if not puts.empty else expirations[0]
            )
            st.caption(f"Expiration: **{exp}** | Total expirations: {len(expirations)}")

            ctab, ptab = st.tabs(["Calls (head)", "Puts (head)"])
            with ctab:
                st.dataframe(calls.head(25), use_container_width=True)
            with ptab:
                st.dataframe(puts.head(25), use_container_width=True)
    except Exception as e:
        st.error(f"Options error: {e!s}")
        st.exception(e)

