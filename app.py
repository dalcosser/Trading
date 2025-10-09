import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta, timezone
import math
import yfinance as yf
import pandas_ta as pta

st.set_page_config(page_title="Codex TA Toolkit", layout="wide")
st.title("📈 Codex TA Toolkit")

# ---------------- Theme / Appearance ----------------
def apply_theme(choice: str) -> str:
    """
    Return plotly template and inject CSS.
    Ensures all white-background surfaces (inputs, expanders, tables, buttons)
    show black text, even in Dark theme.
    """
    css = """
    <style>
      /* ---- Global white-surface readability (sidebar + main) ---- */
      input, select, textarea { color:#000 !important; background:#fff !important; }
      div[role="combobox"] { color:#000 !important; background:#fff !important; border-radius:6px; }
      div[role="combobox"] * { color:#000 !important; background:#fff !important; }
      div[role="listbox"] * { color:#000 !important; background:#fff !important; }

      /* Sidebar specificity */
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] select,
      [data-testid="stSidebar"] textarea { color:#000 !important; background:#fff !important; }
      [data-testid="stSidebar"] div[role="combobox"],
      [data-testid="stSidebar"] div[role="combobox"] * { color:#000 !important; background:#fff !important; }

      /* Number input buttons (+/-) */
      [data-testid="stNumberInput"] button { background:#fff !important; color:#000 !important; border:1px solid #aaa !important; }
      [data-testid="stNumberInput"] svg { fill:#000 !important; color:#000 !important; }

      /* Expanders */
      [data-testid="stExpander"] details,
      [data-testid="stExpander"] summary { background:#fff !important; color:#000 !important; border-radius:6px; }
      [data-testid="stExpander"] * { color:#000 !important; }

      /* Dataframes/tables: black on white to remain readable in Dark */
      .stDataFrame, .stTable { background:#fff !important; color:#000 !important; }
      .stDataFrame [data-testid="stVerticalBlock"], .stTable [data-testid="stVerticalBlock"] { background:#fff !important; color:#000 !important; }
      .stDataFrame th, .stTable th { background:#fff !important; color:#000 !important; }

      /* Tabs header text */
      .stTabs [role="tab"] { color: inherit !important; }

      /* Buttons: white background, black text */
      .stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #aaa !important;
      }
      [data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #aaa !important;
      }
      .stButton > button:hover,
      .stButton > button:active {
        background: #f2f2f2 !important;
        color: #000000 !important;
        border: 1px solid #888 !important;
      }
      .stButton > button:disabled {
        background: #f7f7f7 !important;
        color: #6a6a6a !important;
        border: 1px solid #d0d0d0 !important;
      }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Theme background + font color rules
    if choice == "System":
        st.markdown("""
        <style>
        @media (prefers-color-scheme: dark) {
          html, body, [data-testid="stAppViewContainer"] { background:#0e1117; color:#e6e6e6; }
          [data-testid="stSidebar"] { background:#161a22; }
          [data-testid="stSidebar"] * { color:#e6e6e6; }
        }
        @media (prefers-color-scheme: light) {
          html, body, [data-testid="stAppViewContainer"] { background:#fff; color:#111; }
          [data-testid="stSidebar"] { background:#f6f6f6; }
          [data-testid="stSidebar"] * { color:#111; }
        }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    elif choice == "Dark":
        st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] { background:#0e1117; color:#e6e6e6; }
        [data-testid="stSidebar"] { background:#161a22; }
        [data-testid="stSidebar"] * { color:#e6e6e6; }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    else:
        st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] { background:#fff; color:#111; }
        [data-testid="stSidebar"] { background:#f6f6f6; }
        [data-testid="stSidebar"] * { color:#111; }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_white"


def style_axes(fig: go.Figure, dark: bool, rows: int):
    grid = "#333333" if dark else "#cccccc"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        plot_bgcolor="#000000" if dark else "#ffffff",
        paper_bgcolor="#000000" if dark else "#ffffff",
        font=dict(color="#e6e6e6" if dark else "#111111"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#e6e6e6" if dark else "#111111")
        )
    )
    for r in range(1, rows + 1):
        fig.update_xaxes(row=r, col=1, showgrid=True, gridcolor=grid, zerolinecolor=grid,
                         tickfont_color="#e6e6e6" if dark else "#111111")
        fig.update_yaxes(row=r, col=1, showgrid=True, gridcolor=grid, zerolinecolor=grid,
                         tickfont_color="#e6e6e6" if dark else "#111111")

# ---------------- Sidebar (shared) ----------------
with st.sidebar:
    st.markdown("### Backtesting")
    enable_backtest = st.checkbox("Enable Backtest", value=False)
    strategy = None
    if enable_backtest:
        strategy = st.selectbox(
            "Strategy",
            [
                "Price crosses above VWAP",
                "Price crosses below VWAP",
                "RSI crosses above 70",
                "RSI crosses below 30",
                "MACD crosses above Signal",
                "MACD crosses below Signal"
            ],
            index=0
        )
    st.header("Controls")
    st.markdown("### Appearance")
    theme = st.selectbox("Theme", ["System", "Dark", "Light"], index=0)
template = apply_theme(theme)

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m", "1m"], index=0)
    intraday = interval in {"1h", "30m", "15m", "5m", "1m"}

    if intraday:
        period = st.selectbox("Period (intraday)", ["1d", "5d", "7d", "14d", "30d"], index=2 if interval == "1m" else 1)
        start = date.today() - timedelta(days=7)  # display only
        end = date.today()
    else:
        period = None
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("Start", value=date.today() - timedelta(days=365))
        with c2:
            end = st.date_input("End", value=date.today())

# Tabs
tab1, tab2 = st.tabs(["📊 Chart", "🧾 Options"])

# ---------------- Chart tab controls ----------------
with st.sidebar:
    st.markdown("### Overlays")
    bb_on = st.checkbox("Bollinger Bands", value=True)
    bb_len = st.number_input("BB Length", value=20, min_value=5, max_value=200, step=1)
    bb_std = st.number_input("BB Std Dev", value=2.0, min_value=0.5, max_value=4.0, step=0.5)

    st.markdown("### Moving Averages")
    with st.expander("SMA / EMA", expanded=True):
        ma_periods = [10, 21, 50, 100, 150, 200]
        col_sma, col_ema = st.columns(2)
        with col_sma:
            st.write("**SMA**")
            sma_selected = [p for p in ma_periods if st.checkbox(f"SMA {p}", value=(p in [21, 50, 200]), key=f"sma_{p}")]
        with col_ema:
            st.write("**EMA**")
            ema_selected = [p for p in ma_periods if st.checkbox(f"EMA {p}", value=(p in [21, 50]), key=f"ema_{p}")]

    st.markdown("### Lower Panels")
    show_rsi = st.checkbox("RSI", value=True)
    rsi_len  = st.number_input("RSI Length", value=14, min_value=2, max_value=200, step=1)
    show_sto = st.checkbox("Stochastic %K/%D", value=False)
    sto_k    = st.number_input("%K Length", value=14, min_value=2, max_value=200, step=1)
    sto_d    = st.number_input("%D Length", value=3,  min_value=1, max_value=50,  step=1)
    sto_smooth = st.number_input("%K Smoothing", value=3, min_value=1, max_value=50, step=1)

    st.markdown("### MACD")
    show_macd = st.checkbox("MACD", value=False)
    macd_fast = st.number_input("MACD Fast EMA", value=12, min_value=2, max_value=100, step=1)
    macd_slow = st.number_input("MACD Slow EMA", value=26, min_value=2, max_value=100, step=1)
    macd_signal = st.number_input("MACD Signal EMA", value=9, min_value=1, max_value=50, step=1)

    st.markdown("### Support/Resistance")
    show_sr = st.checkbox("Support/Resistance", value=False)
    sr_lookback = st.number_input("SR Lookback (bars)", value=50, min_value=5, max_value=500, step=5)

    st.markdown("### Candlestick Patterns")
    show_patterns = st.checkbox("Candlestick Patterns", value=False)

    st.markdown("### Anchored VWAP")
    show_vwap = st.checkbox("Anchored VWAP", value=False)
    vwap_anchor = None
    if show_vwap:
        if intraday:
            vwap_anchor = st.text_input("VWAP Anchor (YYYY-MM-DD HH:MM)", value="")
        else:
            vwap_anchor = st.date_input("VWAP Anchor Date", value=start)

    st.markdown("### Chart Layout")
    base_height = st.slider("Chart height (px)", 600, 1600, 1050, 50)

    show_price_labels = st.checkbox("Show price labels (right)", value=False)

    run_chart = st.button("Run Chart", use_container_width=True)

# ---------------- Helpers (chart) ----------------
TARGETS = {"open", "high", "low", "close", "adj close", "adj_close", "adjclose", "volume"}

# --- Anchored VWAP ---
def anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0, price_col: str = "Close", vol_col: str = "Volume") -> pd.Series:
    """
    Calculate Anchored VWAP from a given anchor index (row number).
    Returns a Series with NaN before anchor, and VWAP from anchor forward.
    """
    if price_col not in df.columns or vol_col not in df.columns:
        raise ValueError(f"Columns {price_col} and {vol_col} must be in DataFrame")
    v = df[vol_col].astype(float)
    p = df[price_col].astype(float)
    vwap = pd.Series(np.nan, index=df.index)
    if anchor_idx < 0 or anchor_idx >= len(df):
        return vwap
    cum_vol = v.iloc[anchor_idx:].cumsum()
    cum_pv = (p.iloc[anchor_idx:] * v.iloc[anchor_idx:]).cumsum()
    vwap.iloc[anchor_idx:] = cum_pv / cum_vol
    return vwap

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        for attempt in (df.columns, df.columns.swaplevel()):
            for lvl in range(attempt.nlevels):
                lower_vals = [str(x).lower() for x in attempt.get_level_values(lvl)]
                hits = sum(1 for v in lower_vals if v in TARGETS)
                if hits >= 3:
                    out = df.copy()
                    out.columns = [str(x).title() for x in attempt.get_level_values(lvl)]
                    return out
        out = df.copy()
        out.columns = [str(x).title() for x in df.columns.get_level_values(-1)]
        return out
    out = df.copy()
    out.columns = [str(c).title() for c in out.columns]
    return out

def pick_close_key(cols) -> str | None:
    candidates = {c.lower(): c for c in cols}
    for key in ("close", "adj close", "adj_close", "adjclose"):
        if key in candidates:
            return candidates[key]
    return None

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean().rename(f"SMA({n})")

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean().rename(f"EMA({n})")

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals.fillna(50.0).rename(f"RSI({length})")

def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_len: int = 14, d_len: int = 3, smooth_k: int = 3):
    lowest = low.rolling(k_len, min_periods=1).min()
    highest = high.rolling(k_len, min_periods=1).max()
    k = ((close - lowest) / (highest - lowest).replace(0, np.nan) * 100.0).fillna(50.0)
    if smooth_k > 1:
        k = k.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(d_len, min_periods=1).mean()
    return k.rename(f"%K({k_len})"), d.rename(f"%D({d_len})")

def bbands(series: pd.Series, length: int = 20, n_std: float = 2.0):
    mid = series.rolling(length, min_periods=1).mean()
    sd = series.rolling(length, min_periods=1).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return lower.rename(f"BB Lower({length},{n_std})"), mid.rename(f"BB Mid({length})"), upper.rename(f"BB Upper({length},{n_std})")
    
# --- MACD ---
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.rename("MACD"), signal_line.rename("Signal"), hist.rename("Hist")
    
# --- Support/Resistance ---
def find_support_resistance(series: pd.Series, lookback: int = 50):
    # Simple local min/max finder
    supports = []
    resistances = []
    for i in range(lookback, len(series) - lookback):
        window = series[i - lookback:i + lookback + 1]
        if series[i] == window.min():
            supports.append((series.index[i], series[i]))
        if series[i] == window.max():
            resistances.append((series.index[i], series[i]))
    return supports, resistances
    
# --- Candlestick Patterns ---
import pandas_ta as pta
def detect_patterns(df: pd.DataFrame):
    patterns = {}
    # Use a few common patterns
    for pat in ["cdl_hammer", "cdl_engulfing", "cdl_doji", "cdl_morningstar", "cdl_shootingstar"]:
        try:
            patterns[pat] = getattr(pta, pat)(df["Open"], df["High"], df["Low"], df["Close"])
        except Exception:
            patterns[pat] = pd.Series([0]*len(df), index=df.index)
    return patterns

def infer_pad_timedelta(interval: str) -> timedelta:
    mapping = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "30m": timedelta(minutes=30),
               "15m": timedelta(minutes=15), "5m": timedelta(minutes=5), "1m": timedelta(minutes=1)}
    return mapping.get(interval, timedelta(days=1))

def extend_right_edge(fig: go.Figure, last_ts, interval: str, rows: int):
    pad = infer_pad_timedelta(interval) * 3
    pad_x = last_ts + pad
    for r in range(1, rows + 1):
        fig.add_trace(go.Scatter(x=[pad_x], y=[None], mode="markers",
                                 marker_opacity=0, showlegend=False, hoverinfo="skip"), row=r, col=1)

# ---------------- CHART TAB ----------------
with tab1:
            # --- Simple Backtest Logic ---
            def backtest_price_crosses_vwap(price: pd.Series, vwap: pd.Series):
                signals = (price > vwap) & (price.shift(1) <= vwap.shift(1))
                trades = []
                in_trade = False
                entry_idx = None
                for i in range(1, len(price)):
                    if signals.iloc[i] and not in_trade:
                        entry_idx = i
                        in_trade = True
                    elif in_trade and (price.iloc[i] < vwap.iloc[i]):
                        trades.append({
                            'entry_time': price.index[entry_idx],
                            'entry_price': price.iloc[entry_idx],
                            'exit_time': price.index[i],
                            'exit_price': price.iloc[i],
                            'pnl': price.iloc[i] - price.iloc[entry_idx]
                        })
                        in_trade = False
                        entry_idx = None
                return trades

            # Run backtest if enabled and strategy is selected
            trades = []
            if enable_backtest and strategy == "Price crosses above VWAP" and show_vwap and vwap_idx is not None:
                vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                trades = backtest_price_crosses_vwap(close, vwap_series)
                # Plot buy/sell markers
                for t in trades:
                    fig.add_trace(go.Scatter(
                        x=[t['entry_time']], y=[t['entry_price']],
                        mode="markers", marker_symbol="triangle-up", marker_color="#00ff00", marker_size=14,
                        name="Buy"
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=[t['exit_time']], y=[t['exit_price']],
                        mode="markers", marker_symbol="triangle-down", marker_color="#ff0000", marker_size=14,
                        name="Sell"
                    ), row=1, col=1)
                # Show summary table
                if trades:
                    trade_df = pd.DataFrame(trades)
                    trade_df['holding_period'] = (trade_df['exit_time'] - trade_df['entry_time']).astype(str)
                    st.subheader("Backtest Trades")
                    st.dataframe(trade_df[['entry_time','entry_price','exit_time','exit_price','pnl','holding_period']], use_container_width=True)
                    st.write(f"**Total Trades:** {len(trade_df)}  |  **Total P&L:** {trade_df['pnl'].sum():.2f}  |  **Win Rate:** {100*sum(trade_df['pnl']>0)/len(trade_df):.1f}%")
   if not run_chart:
    st.info("Choose settings in the sidebar and click **Run**.")
    st.stop()
    else:
        try:
            if intraday:
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            else:
                df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), interval=interval, progress=False, auto_adjust=False)

            if df is None or df.empty:
                st.warning("No data returned. Try a different ticker/interval/period.")
                st.stop()

            df = normalize_ohlcv(df)
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Determine VWAP anchor index if enabled
            vwap_idx = None
            if show_vwap and vwap_anchor:
                try:
                    if intraday:
                        # Try to parse as datetime string
                        anchor_dt = pd.to_datetime(vwap_anchor)
                        vwap_idx = df.index.get_loc(anchor_dt, method='nearest') if anchor_dt in df.index else None
                    else:
                        # Date only
                        anchor_dt = pd.to_datetime(vwap_anchor)
                        vwap_idx = df.index.get_loc(anchor_dt, method='nearest') if anchor_dt in df.index else None
                except Exception:
                    vwap_idx = None


            # Count rows for subplots (MACD, RSI, Stoch)
            rows = 2 + (1 if show_rsi else 0) + (1 if show_sto else 0) + (1 if show_macd else 0)
            row_heights = [0.55, 0.13]
            if show_rsi:
                row_heights.append(0.09)
            if show_sto:
                row_heights.append(0.09)
            if show_macd:
                row_heights.append(0.09)
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)



            # Main candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df.get("Open"), high=df.get("High"),
                low=df.get("Low"), close=df.get("Close"), name="OHLC"
            ), row=1, col=1)

            # Add price labels to right side if enabled
            def add_price_label(trace_name, y_val, color):
                fig.add_annotation(
                    xref="paper", x=1.01, y=y_val,
                    xanchor="left", yanchor="middle",
                    text=f"{trace_name}: {y_val:.2f}",
                    font=dict(color=color, size=13),
                    showarrow=False, align="left",
                    bgcolor="#222" if template=="plotly_dark" else "#fff",
                    bordercolor=color, borderwidth=1, opacity=0.95
                )

            if show_price_labels:
                # Candlestick close
                if len(close) > 0:
                    add_price_label("Close", close.iloc[-1], "#00bfff")
                # Anchored VWAP
                if show_vwap and vwap_idx is not None:
                    vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                    last_vwap = vwap_series.dropna().iloc[-1] if vwap_series.dropna().size > 0 else None
                    if last_vwap is not None:
                        add_price_label("VWAP", last_vwap, "#ff9900")
                # SMA/EMA
                for n in sma_selected:
                    s = sma(close, int(n))
                    if len(s.dropna()) > 0:
                        add_price_label(f"SMA({n})", s.dropna().iloc[-1], "#8888ff")
                for n in ema_selected:
                    e = ema(close, int(n))
                    if len(e.dropna()) > 0:
                        add_price_label(f"EMA({n})", e.dropna().iloc[-1], "#ff88ff")

            # Plot anchored VWAP if enabled
            if show_vwap and vwap_idx is not None:
                vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode="lines", name="Anchored VWAP", line=dict(color="#ff9900", width=2, dash="dash")), row=1, col=1)

            fig.add_trace(go.Candlestick(
                x=df.index, open=df.get("Open"), high=df.get("High"),
                low=df.get("Low"), close=df.get("Close"), name="OHLC"
            ), row=1, col=1)



            close_key = pick_close_key(df.columns)
            if close_key is None:
                st.error(f"No usable 'Close' found after normalization. Columns: {list(df.columns)}")
                st.stop()
            close = df[close_key].astype(float)

            # Add price labels to right side if enabled
            def add_price_label(trace_name, y_val, color):
                fig.add_annotation(
                    xref="paper", x=1.01, y=y_val,
                    xanchor="left", yanchor="middle",
                    text=f"{trace_name}: {y_val:.2f}",
                    font=dict(color=color, size=13),
                    showarrow=False, align="left",
                    bgcolor="#222" if template=="plotly_dark" else "#fff",
                    bordercolor=color, borderwidth=1, opacity=0.95
                )

            if show_price_labels:
                # Candlestick close
                if len(close) > 0:
                    add_price_label("Close", close.iloc[-1], "#00bfff")
                # Anchored VWAP
                if show_vwap and vwap_idx is not None:
                    vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                    last_vwap = vwap_series.dropna().iloc[-1] if vwap_series.dropna().size > 0 else None
                    if last_vwap is not None:
                        add_price_label("VWAP", last_vwap, "#ff9900")
                # SMA/EMA
                for n in sma_selected:
                    s = sma(close, int(n))
                    if len(s.dropna()) > 0:
                        add_price_label(f"SMA({n})", s.dropna().iloc[-1], "#8888ff")
                for n in ema_selected:
                    e = ema(close, int(n))
                    if len(e.dropna()) > 0:
                        add_price_label(f"EMA({n})", e.dropna().iloc[-1], "#ff88ff")

            # Add price labels to right side if enabled
            def add_price_label(trace_name, y_val, color):
                fig.add_annotation(
                    xref="paper", x=1.01, y=y_val,
                    xanchor="left", yanchor="middle",
                    text=f"{trace_name}: {y_val:.2f}",
                    font=dict(color=color, size=13),
                    showarrow=False, align="left",
                    bgcolor="#222" if template=="plotly_dark" else "#fff",
                    bordercolor=color, borderwidth=1, opacity=0.95
                )

            if show_price_labels:
                # Candlestick close
                if len(close) > 0:
                    add_price_label("Close", close.iloc[-1], "#00bfff")
                # Anchored VWAP
                if show_vwap and vwap_idx is not None:
                    vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                    last_vwap = vwap_series.dropna().iloc[-1] if vwap_series.dropna().size > 0 else None
                    if last_vwap is not None:
                        add_price_label("VWAP", last_vwap, "#ff9900")
                # SMA/EMA
                for n in sma_selected:
                    s = sma(close, int(n))
                    if len(s.dropna()) > 0:
                        add_price_label(f"SMA({n})", s.dropna().iloc[-1], "#8888ff")
                for n in ema_selected:
                    e = ema(close, int(n))
                    if len(e.dropna()) > 0:
                        add_price_label(f"EMA({n})", e.dropna().iloc[-1], "#ff88ff")

            if bb_on:
                lb, mb, ub = bbands(close, int(bb_len), float(bb_std))
                fig.add_trace(go.Scatter(x=df.index, y=lb, mode="lines", name=lb.name), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=mb, mode="lines", name=mb.name), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=ub, mode="lines", name=ub.name), row=1, col=1)
                
                # Support/Resistance
                if show_sr:
                    supports, resistances = find_support_resistance(close, int(sr_lookback))
                    for t, y in supports:
                        fig.add_hline(y=y, line_dash="dot", line_color="#2ecc40", annotation_text="Support", annotation_position="left", row=1, col=1)
                    for t, y in resistances:
                        fig.add_hline(y=y, line_dash="dot", line_color="#ff4136", annotation_text="Resistance", annotation_position="left", row=1, col=1)
                
                # Candlestick Patterns
                if show_patterns:
                    patterns = detect_patterns(df)
                    for pat, ser in patterns.items():
                        # Markers for detected patterns
                        pat_idx = ser[ser != 0].index
                        fig.add_trace(go.Scatter(x=pat_idx, y=close.loc[pat_idx], mode="markers", marker_symbol="star", marker_size=12, name=pat), row=1, col=1)

            for n in sma_selected:
                s = sma(close, int(n))
                fig.add_trace(go.Scatter(x=df.index, y=s, mode="lines", name=s.name), row=1, col=1)
            for n in ema_selected:
                e = ema(close, int(n))
                fig.add_trace(go.Scatter(x=df.index, y=e, mode="lines", name=e.name), row=1, col=1)

            if "Volume" in df.columns and df["Volume"].notna().any():
                vol_colors = np.where(close >= df.get("Open", close), "rgba(0,200,0,0.6)", "rgba(200,0,0,0.6)")
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, showlegend=False), row=2, col=1)
            else:
                fig.add_trace(go.Bar(x=df.index, y=[0]*len(df), name="Volume", showlegend=False), row=2, col=1)

            next_row = 3
            if show_rsi:
                r = rsi(close, int(rsi_len))
                fig.add_trace(go.Scatter(x=df.index, y=r, mode="lines", name=r.name), row=next_row, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#555", row=next_row, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#555", row=next_row, col=1)
                fig.update_yaxes(range=[0, 100], row=next_row, col=1, title_text="RSI")
                next_row += 1

            if show_sto:
                k, d = stoch_kd(df["High"], df["Low"], close, int(sto_k), int(sto_d), int(sto_smooth))
                fig.add_trace(go.Scatter(x=df.index, y=k, mode="lines", name=k.name), row=next_row, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=d, mode="lines", name=d.name), row=next_row, col=1)
                fig.add_hline(y=20, line_dash="dot", line_color="#555", row=next_row, col=1)
                fig.add_hline(y=80, line_dash="dot", line_color="#555", row=next_row, col=1)
                fig.update_yaxes(range=[0, 100], row=next_row, col=1, title_text="Stoch")
                
                if show_macd:
                    macd_line, signal_line, hist = macd(close, int(macd_fast), int(macd_slow), int(macd_signal))
                    fig.add_trace(go.Scatter(x=df.index, y=macd_line, mode="lines", name="MACD"), row=next_row, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=signal_line, mode="lines", name="Signal"), row=next_row, col=1)
                    fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist", marker_color=np.where(hist>=0, "#2ecc40", "#ff4136")), row=next_row, col=1)
                    fig.update_yaxes(title_text="MACD", row=next_row, col=1)
                    next_row += 1

            last_ts = pd.to_datetime(df.index[-1])
            extend_right_edge(fig, last_ts, interval, rows)

            fig.update_layout(title=f"{ticker} — {interval}", xaxis_rangeslider_visible=False, height=base_height)
            style_axes(fig, dark=(template == "plotly_dark"), rows=rows)

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Rows: {len(df)} | Columns: {list(df.columns)}")

        except Exception as e:
            st.error(f"Error fetching or plotting data: {e}")

# ---------------- OPTIONS TAB ----------------

# Lightweight BS delta (no scipy)
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
def _phi(x: float) -> float:  # pdf
    return math.exp(-0.5 * x * x) / SQRT2PI
def _Phi(x: float) -> float:  # cdf via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, q: float, kind: str) -> float:
    """Compute approximate Black-Scholes delta (no scipy)."""
    try:
        if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
            return math.nan
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        if kind == "call":
            return math.exp(-q * T) * 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        else:
            return -math.exp(-q * T) * 0.5 * (1.0 - math.erf(d1 / math.sqrt(2.0)))
    except Exception:
        return math.nan

        return float("nan")

@st.cache_data(show_spinner=False, ttl=300)
def fetch_expirations(ticker: str) -> list[str]:
    t = yf.Ticker(ticker)
    return t.options or []

@st.cache_data(show_spinner=False, ttl=300)
def fetch_chain(ticker: str, expiration: str):
    t = yf.Ticker(ticker)
    oc = t.option_chain(expiration)
    calls = oc.calls.copy()
    puts = oc.puts.copy()
    # normalize column names
    calls.columns = [str(c).replace(' ', '_').lower() for c in calls.columns]
    puts.columns  = [str(c).replace(' ', '_').lower()  for c in puts.columns]
    return calls, puts

def spot_price(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="1d")
        if not h.empty:
            return float(h["Close"][-1])
    except Exception:
        pass
    return None

with tab2:
    st.subheader("Options Chain")
    # Config section for greeks & charts
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([1,1,1,2])
    with cfg_col1:
        r_rate = st.number_input("Risk-free r (decimal)", value=0.02, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")
    with cfg_col2:
        q_div  = st.number_input("Dividend q (decimal)", value=0.0, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")
    with cfg_col3:
        show_oi_chart  = st.checkbox("OI by Strike", value=True)
        show_vol_chart = st.checkbox("Vol by Strike", value=True)
        show_iv_chart  = st.checkbox("IV (Smile)", value=True)
    with cfg_col4:
        load_opts = st.button("Load / Refresh Chain", use_container_width=True)

    if not load_opts and "opts_expirations" not in st.session_state:
        st.info("Click **Load / Refresh Chain** to fetch expirations and the nearest chain.")
    else:
        try:
            if load_opts or "opts_expirations" not in st.session_state:
                exps = fetch_expirations(ticker)
                if not exps:
                    st.warning("No options data available for this ticker.")
                    st.stop()
                st.session_state["opts_expirations"] = exps

                # nearest expiry (>= today)
                today = datetime.now(timezone.utc).date()
                nearest = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days))
                st.session_state["opts_selected_exp"] = nearest

            exps = st.session_state["opts_expirations"]
            sel = st.selectbox("Expiration", options=exps, index=exps.index(st.session_state["opts_selected_exp"]))
            st.session_state["opts_selected_exp"] = sel

            with st.spinner(f"Fetching chain for {sel}..."):
                calls, puts = fetch_chain(ticker, sel)

            # Add delta using BS with yfinance implied volatility if available
            S = spot_price(ticker)
            exp_dt = datetime.strptime(sel, "%Y-%m-%d").date()
            T = max((exp_dt - datetime.now(timezone.utc).date()).days, 0) / 365.0

            def add_delta(df: pd.DataFrame, kind: str) -> pd.DataFrame:
                out = df.copy()
                if S is None or T <= 0 or "implied_volatility" not in out.columns or "strike" not in out.columns:
                    out["delta"] = np.nan
                    return out
                iv = pd.to_numeric(out["implied_volatility"], errors="coerce").fillna(np.nan).astype(float)
                strikes = pd.to_numeric(out["strike"], errors="coerce").astype(float)
                deltas = []
                for k, sig in zip(strikes, iv):
                    deltas.append(bs_delta(S, float(k), float(T), float(r_rate), float(sig or 0.0), float(q_div), kind))
                out["delta"] = deltas
                return out

            calls = add_delta(calls, "call")
            puts  = add_delta(puts,  "put")

            # Filters
            f1, f2, f3 = st.columns(3)
            with f1:
                min_oi = st.number_input("Min Open Interest", value=0, min_value=0, step=10)
            with f2:
                min_vol = st.number_input("Min Volume", value=0, min_value=0, step=10)
            with f3:
                moneyness = st.selectbox("Moneyness", ["All", "OTM", "ATM (±1%)", "ITM"])

            # moneyness tagging (uses spot S if available)
            def tag_filter(df: pd.DataFrame, kind: str) -> pd.DataFrame:
                out = df.copy()
                if S is not None and "strike" in out.columns:
                    out["moneyness"] = (out["strike"] - S) / S
                    if kind == "call":
                        itm_mask = out["strike"] < S
                    else:
                        itm_mask = out["strike"] > S
                    if moneyness == "OTM":
                        out = out[~itm_mask]
                    elif moneyness == "ITM":
                        out = out[itm_mask]
                    elif moneyness == "ATM (±1%)":
                        out = out[abs(out["moneyness"]) <= 0.01]
                if "open_interest" in out.columns:
                    out = out[out["open_interest"].fillna(0) >= min_oi]
                if "volume" in out.columns:
                    out = out[out["volume"].fillna(0) >= min_vol]
                return out

            cc = tag_filter(calls, "call").sort_values("strike")
            pp = tag_filter(puts,  "put").sort_values("strike")

            # Tables (black on white ensured by global CSS)
            t1, t2 = st.tabs(["Calls", "Puts"])
            with t1:
                st.dataframe(cc, use_container_width=True)
            with t2:
                st.dataframe(pp, use_container_width=True)

            # Charts vs strike
            def add_vline(fig, xval, label):
                if xval is None:
                    return
                fig.add_vline(x=xval, line_dash="dash", line_color="#888", annotation_text=label, annotation_position="top")

            if show_oi_chart:
                fig_oi = go.Figure()
                if "open_interest" in cc.columns:
                    fig_oi.add_trace(go.Scatter(x=cc["strike"], y=cc["open_interest"], mode="lines+markers", name="Calls OI"))
                if "open_interest" in pp.columns:
                    fig_oi.add_trace(go.Scatter(x=pp["strike"], y=pp["open_interest"], mode="lines+markers", name="Puts OI"))
                add_vline(fig_oi, S, "Spot")
                fig_oi.update_layout(title=f"Open Interest by Strike — {sel}", xaxis_title="Strike", yaxis_title="Open Interest", height=380)
                st.plotly_chart(fig_oi, use_container_width=True)

            if show_vol_chart:
                fig_vol = go.Figure()
                if "volume" in cc.columns:
                    fig_vol.add_trace(go.Scatter(x=cc["strike"], y=cc["volume"], mode="lines+markers", name="Calls Volume"))
                if "volume" in pp.columns:
                    fig_vol.add_trace(go.Scatter(x=pp["strike"], y=pp["volume"], mode="lines+markers", name="Puts Volume"))
                add_vline(fig_vol, S, "Spot")
                fig_vol.update_layout(title=f"Volume by Strike — {sel}", xaxis_title="Strike", yaxis_title="Volume", height=380)
                st.plotly_chart(fig_vol, use_container_width=True)

            if show_iv_chart:
                fig_iv = go.Figure()
                if "implied_volatility" in cc.columns:
                    fig_iv.add_trace(go.Scatter(x=cc["strike"], y=cc["implied_volatility"]*100, mode="lines+markers", name="Calls IV%"))
                if "implied_volatility" in pp.columns:
                    fig_iv.add_trace(go.Scatter(x=pp["strike"], y=pp["implied_volatility"]*100, mode="lines+markers", name="Puts IV%"))
                add_vline(fig_iv, S, "Spot")
                fig_iv.update_layout(title=f"Implied Volatility (Smile) — {sel}", xaxis_title="Strike", yaxis_title="IV (%)", height=380)
                st.plotly_chart(fig_iv, use_container_width=True)

            # theme for options charts (match main)
            for f in ["fig_oi", "fig_vol", "fig_iv"]:
                pass  # (plotly template is fine; page-wide colors already set)

            if S is not None:
                st.caption(f"Spot ≈ {S:.2f} | Expirations: {len(exps)} | Showing: {sel}")
            else:
                st.caption(f"Spot unavailable | Expirations: {len(exps)} | Showing: {sel}")

        except Exception as e:
            st.error(f"Options error: {e}")
