import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta, timezone
import math
import yfinance as yf
import time
import os
import json
from typing import Optional
try:
    import pandas_ta as pta  # for candlestick pattern detection
    HAS_PANDAS_TA = True
except Exception:
    pta = None
    HAS_PANDAS_TA = False

st.set_page_config(page_title="Codex TA Toolkit", layout="wide")
st.title("Codex TA Toolkit")

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

      /* Ensure TradingView embed uses full width */
      .tradingview-widget-container, .tradingview-widget-container__widget {
        width: 100% !important;
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
    st.header("Controls")

# Allow overriding the per-ticker Parquet directory from the UI for reliability
with st.sidebar:
    try:
        default_parquet_dir = os.environ.get('PER_TICKER_PARQUET_DIR') or f"C:/Users/{os.environ.get('USERNAME','')}/Documents/Visual Code/Polygon Data/per_ticker_daily"
    except Exception:
        default_parquet_dir = ""
    parquet_dir_input = st.text_input("Per-ticker Parquet directory", value=default_parquet_dir)
    if parquet_dir_input and parquet_dir_input != os.environ.get('PER_TICKER_PARQUET_DIR'):
        os.environ['PER_TICKER_PARQUET_DIR'] = parquet_dir_input
        try:
            _autofind_parquet_path.clear()
        except Exception:
            pass

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
tab1, tab2, tab3 = st.tabs(["Chart", "Options", "TradingView"])

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
    lower_options = ["Volume", "RSI", "Stochastic %K/%D", "MACD (panel)"]
    lower_selected = st.multiselect("Add panels", options=lower_options, default=["Volume", "RSI"])
    show_volume = "Volume" in lower_selected
    show_rsi = "RSI" in lower_selected
    rsi_len  = st.number_input("RSI Length", value=14, min_value=2, max_value=200, step=1, disabled=not show_rsi)
    show_sto = "Stochastic %K/%D" in lower_selected
    sto_k    = st.number_input("%K Length", value=14, min_value=2, max_value=200, step=1, disabled=not show_sto)
    sto_d    = st.number_input("%D Length", value=3,  min_value=1, max_value=50,  step=1, disabled=not show_sto)
    sto_smooth = st.number_input("%K Smoothing", value=3, min_value=1, max_value=50, step=1, disabled=not show_sto)

    st.markdown("### MACD Settings")
    show_macd = "MACD (panel)" in lower_selected
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

    force_fresh = st.checkbox("Force fresh fetch (bypass cache)", value=False)
    try:
        st.session_state['force_refresh'] = force_fresh
    except Exception:
        pass
    use_polygon = st.checkbox("Use Polygon if key present", value=True)
    polygon_key_input = st.text_input("Polygon API Key (optional)", value="", type="password")
    st.session_state['use_polygon'] = use_polygon
    if polygon_key_input:
        st.session_state['polygon_api_key'] = polygon_key_input.strip()

    # Diagnostics (key detection + last source)
    try:
        _poly_key = st.session_state.get('polygon_api_key')
        if not _poly_key:
            try:
                if hasattr(st, 'secrets') and ('POLYGON_API_KEY' in st.secrets):
                    _poly_key = st.secrets['POLYGON_API_KEY']
            except Exception:
                pass
        if not _poly_key:
            _poly_key = os.getenv('POLYGON_API_KEY')
        _last_src = st.session_state.get('last_fetch_provider', '-')
        st.markdown("### Diagnostics")
        st.caption(f"Polygon key detected: {bool(_poly_key)} | Last source: {_last_src}")
    except Exception:
        pass

# ---------------- Helpers (chart) ----------------
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

    st.markdown("### Appearance")
    theme = st.selectbox("Theme", ["System", "Dark", "Light"], index=0)
template = apply_theme(theme)

# ---------------- Helpers (chart) ----------------
TARGETS = {"open", "high", "low", "close", "adj close", "adj_close", "adjclose", "volume"}

# --- Data fetch fallback helpers ---
def best_period_for(interval_str: str, desired: Optional[str]) -> str:
    if interval_str == "1m":
        return "7d"
    if interval_str in {"5m", "15m", "30m", "60m", "1h"}:
        return desired if desired in {"5d", "7d", "14d", "30d", "60d"} else "30d"
    return desired or "1y"

# --- TradingView symbol mapper (heuristic) ---
def tv_symbol_for(sym: str) -> str:
    s = (sym or "").strip().upper()
    if ":" in s:
        return s  # already namespaced
    # Common indices
    index_map = {
        "^GSPC": "SP:SPX",
        "SPX": "SP:SPX",
        "^NDX": "NASDAQ:NDX",
        "NDX": "NASDAQ:NDX",
        "^DJI": "DJ:DJI",
        "DJI": "DJ:DJI",
        "^VIX": "CBOE:VIX",
        "VIX": "CBOE:VIX",
    }
    if s in index_map:
        return index_map[s]
    if s.startswith("^"):
        return s[1:]
    # Common futures roots to TradingView continuous front contract
    fut_map = {
        "ES": "CME_MINI:ES1!", "NQ": "CME_MINI:NQ1!", "YM": "CBOT_MINI:YM1!", "RTY": "CME:RTY1!",
        "CL": "NYMEX:CL1!", "NG": "NYMEX:NG1!", "RB": "NYMEX:RB1!", "HO": "NYMEX:HO1!",
        "GC": "COMEX:GC1!", "SI": "COMEX:SI1!", "HG": "COMEX:HG1!",
        "ZC": "CBOT:ZC1!", "ZS": "CBOT:ZS1!", "ZW": "CBOT:ZW1!", "ZM": "CBOT:ZM1!", "ZL": "CBOT:ZL1!",
        "KC": "ICEUS:KC1!", "SB": "ICEUS:SB1!", "CC": "ICEUS:CC1!", "CT": "ICEUS:CT1!", "OJ": "ICEUS:OJ1!",
    }
    root = s.split("=")[0]
    if root in fut_map:
        return fut_map[root]
    # Default to NASDAQ namespace for plain equities; users can change in-widget
    if s.isalpha() and 1 <= len(s) <= 5:
        return f"NASDAQ:{s}"
    return s

# --- Symbol normalization (indices and futures continuous contracts) ---
FUTURES_CONTINUOUS = {
    # Equity index futures (CME)
    "ES", "NQ", "YM", "RTY",
    # Rates (CME)
    "ZN", "ZB", "ZF", "ZT",
    # Energies (NYMEX)
    "CL", "NG", "RB", "HO", "BZ", "BRN",
    # Metals (COMEX)
    "GC", "SI", "HG", "PA", "PL",
    # Ags (CBOT)
    "ZC", "ZS", "ZW", "ZM", "ZL",
    # Softs (ICE)
    "KC", "SB", "CC", "CT", "OJ",
}

def normalize_input_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if s.startswith("^"):
        return s  # caret indices handled downstream
    if "=F" in s or ":" in s:
        return s  # already explicit
    root = s.split()[0]
    if root in FUTURES_CONTINUOUS:
        return root + "=F"  # map to Yahoo continuous contract
    return s

# --- Futures specific-contract parser (e.g., ESZ24, CLX2024) ---
_MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

_FUTURES_SUFFIX = {
    # Exchange code suffix as used by Yahoo
    # CME group equity index
    'ES': 'CME', 'NQ': 'CME', 'RTY': 'CME', 'YM': 'CBT',
    # Treasuries (CBOT)
    'ZN': 'CBT', 'ZB': 'CBT', 'ZF': 'CBT', 'ZT': 'CBT',
    # Energies (NYMEX)
    'CL': 'NYM', 'NG': 'NYM', 'RB': 'NYM', 'HO': 'NYM', 'BZ': 'NYM', 'BRN': 'NYM',
    # Metals (COMEX/NYMEX)
    'GC': 'CMX', 'SI': 'CMX', 'HG': 'CMX', 'PA': 'NYM', 'PL': 'NYM',
    # Ags (CBOT)
    'ZC': 'CBT', 'ZS': 'CBT', 'ZW': 'CBT', 'ZM': 'CBT', 'ZL': 'CBT',
    # Softs (ICE US)
    'KC': 'NYB', 'SB': 'NYB', 'CC': 'NYB', 'CT': 'NYB', 'OJ': 'NYB',
}

def build_futures_contract_candidates(sym: str):
    s = (sym or '').strip().upper()
    # Pattern: ROOT + MONTH_LETTER + YY or YYYY, e.g., ESZ24, CLX2024
    import re
    m = re.fullmatch(r"([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{2}|\d{4})", s)
    if not m:
        return None
    root, mon, year = m.group(1), m.group(2), m.group(3)
    yy = year[-2:]
    yyyy = ("20" + yy) if len(year) == 2 else year
    suffix = _FUTURES_SUFFIX.get(root)
    candidates = []
    # Yahoo with exchange suffix
    if suffix:
        candidates.append(f"{root}{mon}{yy}.{suffix}")
    # Yahoo without suffix
    candidates.append(f"{root}{mon}{yy}")
    # Continuous fallback
    candidates.append(f"{root}=F")
    # Polygon-style candidates (best-effort)
    candidates.append(f"C:{root}{mon}{yyyy}")
    candidates.append(f"{root}{mon}{yyyy}")
    return candidates

def _fetch_ohlc_uncached(
    ticker: str,
    *,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    import pandas as _pd
    from time import sleep as _sleep

    # Normalize interval for providers (Yahoo prefers '60m' over '1h')
    _interval = '60m' if interval == '1h' else interval

    # Collect errors for diagnostics
    try:
        st.session_state['last_fetch_errors'] = []
    except Exception:
        pass

    # Decide order for providers: Polygon -> yfinance -> yahooquery -> Stooq(1d)

    # Helper: Polygon first if configured
    def _try_polygon():
        # Always use Polygon when a key is available (sidebar/session, st.secrets, or env)
        try:
            api_key = st.session_state.get('polygon_api_key')
            if not api_key:
                try:
                    if hasattr(st, 'secrets') and ('POLYGON_API_KEY' in st.secrets):
                        api_key = st.secrets['POLYGON_API_KEY']
                except Exception:
                    pass
            if not api_key:
                api_key = os.getenv('POLYGON_API_KEY')
        except Exception:
            api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return None
        try:
            try:
                from src.data_providers.polygon_fetch import fetch_polygon_ohlc as _poly_fetch  # type: ignore
            except Exception:
                try:
                    from polygon_fetch import fetch_polygon_ohlc as _poly_fetch  # type: ignore
                except Exception:
                    _poly_fetch = None
            if _poly_fetch is None:
                return None
            dfp = _poly_fetch(ticker, interval=_interval, period=period, start=start, end=end, api_key=api_key)
            if dfp is not None and not dfp.empty:
                try:
                    st.session_state['last_fetch_provider'] = 'polygon'
                except Exception:
                    pass
                return dfp
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"polygon: {e}")
            except Exception:
                pass
        return None

    # Helper: yahooquery block
    def _try_yahooquery():
        try:
            from yahooquery import Ticker as _YQTicker
            yq = _YQTicker(ticker)
            if period:
                yq_df = yq.history(period=period, interval=_interval)
            else:
                yq_df = yq.history(start=start, end=end, interval=_interval)
            if yq_df is None or yq_df.empty:
                return None
            # Flatten MultiIndex to DatetimeIndex if needed
            if isinstance(yq_df.index, _pd.MultiIndex):
                if 'date' in yq_df.index.names:
                    yq_df = yq_df.reset_index().set_index('date')
                else:
                    yq_df = yq_df.reset_index()
            if not isinstance(yq_df.index, _pd.DatetimeIndex) and 'date' in yq_df.columns:
                yq_df['date'] = _pd.to_datetime(yq_df['date'])
                yq_df = yq_df.set_index('date')
            rename = {
                "open": "Open","high": "High","low": "Low","close": "Close",
                "adjclose": "Adj Close","adj_close": "Adj Close","volume": "Volume",
            }
            yq_df = yq_df.rename(columns=lambda c: rename.get(str(c).lower(), str(c).title()))
            keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in yq_df.columns]
            if keep:
                yq_df = yq_df[keep]
            try:
                st.session_state['last_fetch_provider'] = 'yahooquery'
            except Exception:
                pass
            return yq_df
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"yahooquery: {e}")
            except Exception:
                pass
            return None

    # 1) Polygon first if available
    poly_df = _try_polygon()
    if isinstance(poly_df, _pd.DataFrame) and not poly_df.empty:
        return poly_df

    # 2) yfinance with small retries
    for attempt in range(3):
        try:
            if period:
                df = yf.download(ticker, period=period, interval=_interval, progress=False, auto_adjust=False, threads=False)
            else:
                df = yf.download(ticker, start=start, end=end, interval=_interval, progress=False, auto_adjust=False, threads=False)
            if isinstance(df, _pd.DataFrame) and not df.empty:
                try:
                    st.session_state['last_fetch_provider'] = 'yfinance'
                except Exception:
                    pass
                return df
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"yfinance attempt {attempt+1} (auto_adjust=False): {e}")
            except Exception:
                pass
        _sleep(0.6 * (attempt + 1))

    # Retry with auto_adjust=True once
    try:
        if period:
            df = yf.download(ticker, period=period, interval=_interval, progress=False, auto_adjust=True, threads=False)
        else:
            df = yf.download(ticker, start=start, end=end, interval=_interval, progress=False, auto_adjust=True, threads=False)
        if isinstance(df, _pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        try:
            st.session_state['last_fetch_errors'].append(f"yfinance (auto_adjust=True): {e}")
        except Exception:
            pass

    # 3) yahooquery fallback
    yq_df = _try_yahooquery()
    if isinstance(yq_df, _pd.DataFrame) and not yq_df.empty:
        if isinstance(yq_df, _pd.DataFrame) and not yq_df.empty:
            try:
                st.session_state['last_fetch_provider'] = 'yahooquery'
            except Exception:
                pass
        return yq_df

    # 3) Stooq daily fallback via pandas-datareader (only for 1d)
    try:
        if _interval == '1d':
            try:
                from pandas_datareader import data as _pdr
            except Exception as e:
                try:
                    st.session_state['last_fetch_errors'].append(f"pandas-datareader not available for Stooq fallback: {e}")
                except Exception:
                    pass
                return _pd.DataFrame()

            # Determine date range
            _start = None
            _end = None
            try:
                if start and end:
                    _start = _pd.to_datetime(start)
                    _end = _pd.to_datetime(end)
                elif period:
                    now = _pd.Timestamp.utcnow().normalize()
                    per_map = {
                        '7d': 7, '14d': 14, '30d': 30, '60d': 60, '90d': 90,
                        '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'max': 36500
                    }
                    days = per_map.get(str(period).lower(), 365)
                    _start = now - _pd.Timedelta(days=days)
                    _end = now + _pd.Timedelta(days=1)
                else:
                    _end = _pd.Timestamp.utcnow().normalize() + _pd.Timedelta(days=1)
                    _start = _end - _pd.Timedelta(days=365)
            except Exception:
                pass

            sym = str(ticker).strip().upper()
            stooq_symbol = sym + '.US' if sym.isalpha() else sym
            try:
                stq = _pdr.DataReader(stooq_symbol, 'stooq', start=_start, end=_end)
                if isinstance(stq, _pd.DataFrame) and not stq.empty:
                    stq = stq.sort_index()
                    # ensure yahoo-like column casing
                    stq = stq.rename(columns={
                        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                    })
                    try:
                        st.session_state['last_fetch_provider'] = 'stooq'
                    except Exception:
                        pass
                    return stq
            except Exception as e:
                try:
                    st.session_state['last_fetch_errors'].append(f"stooq fallback: {e}")
                except Exception:
                    pass
    except Exception:
        pass


@st.cache_data(show_spinner=False, ttl=120)
def _fetch_ohlc_cached(
    ticker: str,
    *,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_buster: Optional[str] = None,
):
    # cache_buster participates in cache key but is unused otherwise
    return _fetch_ohlc_uncached(ticker, interval=interval, period=period, start=start, end=end)


def fetch_ohlc_with_fallback(
    ticker: str,
    *,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Cached fetch that avoids caching empty results by retrying with a cache-buster."""
    try:
        force = bool(st.session_state.get('force_refresh'))
    except Exception:
        force = False
    first_buster = (str(time.time()) if force else None)
    df = _fetch_ohlc_cached(ticker, interval=interval, period=period, start=start, end=end, cache_buster=first_buster)
    try:
        import pandas as _pd
        is_empty = (df is None) or (isinstance(df, _pd.DataFrame) and df.empty)
    except Exception:
        is_empty = df is None
    if not is_empty:
        return df
    # Avoid returning a cached empty; retry once with a unique cache key
    buster = str(time.time())
    return _fetch_ohlc_cached(ticker, interval=interval, period=period, start=start, end=end, cache_buster=buster)

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
def detect_patterns(df: pd.DataFrame):
    patterns = {}
    # If pandas-ta is unavailable, return empty/zero series so the app still works
    if not HAS_PANDAS_TA or pta is None:
        for pat in ["cdl_hammer", "cdl_engulfing", "cdl_doji", "cdl_morningstar", "cdl_shootingstar"]:
            patterns[pat] = pd.Series([0] * len(df), index=df.index)
        return patterns

    # Use a few common patterns (guard each call)
    for pat in ["cdl_hammer", "cdl_engulfing", "cdl_doji", "cdl_morningstar", "cdl_shootingstar"]:
        try:
            patterns[pat] = getattr(pta, pat)(df["Open"], df["High"], df["Low"], df["Close"])
        except Exception:
            patterns[pat] = pd.Series([0] * len(df), index=df.index)
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

# ---------------- Historical Stats (Polygon flat files) ----------------
@st.cache_data(show_spinner=False)
def _normalize_host_path(p: str | None) -> str | None:
    if p is None:
        return None
    try:
        import os, re
        s = str(p).strip().strip('"').strip("'")
        # If running on non-Windows and given a Windows path, map to WSL style
        if os.name != 'nt' and re.match(r'^[a-zA-Z]:[\\/]', s):
            drive = s[0].lower()
            rest = s[2:].replace('\\','/')
            s = f"/mnt/{drive}/{rest.lstrip('/')}"
        return os.path.normpath(s)
    except Exception:
        return p

@st.cache_data(show_spinner=False)
def _autofind_parquet_path(ticker: str) -> str | None:
    """
    Try to locate a per-ticker parquet like <TICKER>.parquet in common folders
    without surfacing paths in the UI.
    Order:
      1) PER_TICKER_PARQUET_DIR env var
      2) Common Windows Documents paths (per_ticker_daily_tech, per_ticker_daily)
      3) WSL equivalents
      4) ./per_ticker_daily_tech or ./per_ticker_daily under CWD
    Returns a normalized path or None.
    """
    import os
    from pathlib import Path
    t = ticker.strip().upper()
    fname = f"{t}.parquet"
    # Also consider alternative filename variants for symbols with punctuation
    alt_names = {fname}
    base = t
    for repl in (('.', '_'), ('/', '_'), ('-', '_'), (' ', '_')):
        base = t.replace(repl[0], repl[1])
        alt_names.add(f"{base}.parquet")
    alt_names.add(f"{t.replace('.', '')}.parquet")
    # 1) Env var
    env_dir = os.environ.get('PER_TICKER_PARQUET_DIR')
    if env_dir:
        # Try exact and alternate names
        for nm in list(alt_names):
            p = _normalize_host_path(os.path.join(env_dir, nm))
            if p and os.path.exists(p):
                return p
    # 2) Windows Documents (standard + 'Visual Code' layout)
    common_dirs = [
        # Standard Documents
        f"C:/Users/{os.environ.get('USERNAME', '')}/Documents/Polygon Data/per_ticker_daily_tech",
        f"C:/Users/{os.environ.get('USERNAME', '')}/Documents/Polygon Data/per_ticker_daily",
        # Visual Code workspace under Documents
        f"C:/Users/{os.environ.get('USERNAME', '')}/Documents/Visual Code/Polygon Data/per_ticker_daily_tech",
        f"C:/Users/{os.environ.get('USERNAME', '')}/Documents/Visual Code/Polygon Data/per_ticker_daily",
    ]
    # 3) WSL equivalents
    common_dirs += [
        f"/mnt/c/Users/{os.environ.get('USERNAME', '').lower()}/Documents/Polygon Data/per_ticker_daily_tech",
        f"/mnt/c/Users/{os.environ.get('USERNAME', '').lower()}/Documents/Polygon Data/per_ticker_daily",
        f"/mnt/c/Users/{os.environ.get('USERNAME', '').lower()}/Documents/Visual Code/Polygon Data/per_ticker_daily_tech",
        f"/mnt/c/Users/{os.environ.get('USERNAME', '').lower()}/Documents/Visual Code/Polygon Data/per_ticker_daily",
    ]
    # 4) Local under CWD
    common_dirs += [
        str(Path.cwd() / 'per_ticker_daily_tech'),
        str(Path.cwd() / 'per_ticker_daily'),
    ]
    for d in common_dirs:
        # Try direct matches first
        for nm in list(alt_names):
            p = _normalize_host_path(os.path.join(d, nm))
            if p and os.path.exists(p):
                return p
        # Fallback: scan directory and match by alphanumeric-only stem
        try:
            from pathlib import Path as _P
            dd = _P(_normalize_host_path(d))
            if dd.exists():
                want = ''.join(ch for ch in t if ch.isalnum())
                for fp in dd.glob('*.parquet'):
                    stem = fp.stem
                    if stem.endswith('.csv'):
                        stem = stem[:-4]
                    key = ''.join(ch for ch in stem.upper() if ch.isalnum())
                    if key == want:
                        return str(fp)
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False)
def _autofind_report_excel_path(ticker: str) -> str | None:
    """
    Try to locate <TICKER>_technicals.xlsx without showing paths in the UI.
    Order:
      1) POLYGON_REPORTS_DIR env var
      2) Common Windows path in user's Documents
      3) WSL-style equivalent under /mnt/c
      4) ./reports relative to CWD
      5) Recursive search under user's Documents Polygon Data
    Returns a normalized path or None.
    """
    import os, glob, getpass
    t = ticker.strip().upper()
    fname = f"{t}_technicals.xlsx"
    # 1) Env var
    env_dir = os.environ.get('POLYGON_REPORTS_DIR')
    if env_dir:
        p = _normalize_host_path(os.path.join(env_dir, fname))
        if p and os.path.exists(p):
            return p
    # 2) Common Windows path
    user = getpass.getuser()
    win_dir = f"C:/Users/{user}/Documents/Polygon Data/reports"
    p = _normalize_host_path(os.path.join(win_dir, fname))
    if p and os.path.exists(p):
        return p
    # 3) WSL equivalent
    wsl_dir = f"/mnt/c/Users/{user}/Documents/Polygon Data/reports"
    p = _normalize_host_path(os.path.join(wsl_dir, fname))
    if p and os.path.exists(p):
        return p
    # 4) ./reports relative to CWD
    local_dir = os.path.join(os.getcwd(), 'reports')
    p = _normalize_host_path(os.path.join(local_dir, fname))
    if p and os.path.exists(p):
        return p
    # 5) Recursive search under user's Documents Polygon Data
    base_dirs = [f"C:/Users/{user}/Documents/Polygon Data", f"/mnt/c/Users/{user}/Documents/Polygon Data"]
    for bd in base_dirs:
        root = _normalize_host_path(bd)
        if root and os.path.exists(root):
            pattern = os.path.join(root, '**', fname)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return _normalize_host_path(matches[0])
    return None

@st.cache_data(show_spinner=False)
def _load_polygon_daily_for_ticker(
    data_root: str,
    ticker: str,
    reports_dir: str | None = None,
    technicals_script: str | None = None,
    auto_generate_report: bool = True,
    excel_override: object | None = None,
    excel_path_override: str | None = None,
    allow_yahoo_fallback: bool = False,
) -> pd.DataFrame:
    """
    Load per-ticker daily parquet from a Polygon flat-files export.
    Expects a file like `<data_root>/per_ticker_daily/<TICKER>.parquet`.
    Normalizes columns to ['Date','Open','High','Low','Close','Volume'] with Date as datetime.
    Returns sorted ascending by Date.
    """
    import os
    import pandas as pd

    t = ticker.strip().upper()
    df = None
    # 0) Prefer per-ticker parquet if present (already contains technicals)
    try:
        pq_path = _autofind_parquet_path(t)
        if pq_path and os.path.exists(pq_path):
            df_pq = pd.read_parquet(pq_path)
            if isinstance(df_pq, pd.DataFrame) and not df_pq.empty:
                # Normalize to expected columns
                cols = {str(c).lower(): c for c in df_pq.columns}
                date_col = cols.get('timestamp') or cols.get('date')
                if date_col is not None:
                    out = pd.DataFrame({'Date': pd.to_datetime(df_pq[date_col], errors='coerce')})
                    for c in ('Open','High','Low','Close','Volume'):
                        src = cols.get(c.lower())
                        if src is not None and src in df_pq.columns:
                            out[c] = pd.to_numeric(df_pq[src], errors='coerce')
                    out = out.dropna(subset=['Date','Close']) if 'Close' in out.columns else out.dropna(subset=['Date'])
                    out = out.sort_values('Date').reset_index(drop=True)
                    return out
    except Exception:
        pass
    # 0) If a file-like Excel was provided (uploaded), parse it first
    if excel_override is not None:
        try:
            try:
                import openpyxl  # noqa: F401
                xl = pd.ExcelFile(excel_override, engine='openpyxl')
            except ImportError:
                xl = pd.ExcelFile(excel_override)
            def _try_sheet_to_ohlc(_df: pd.DataFrame) -> pd.DataFrame | None:
                d = _df.copy()
                try:
                    if not isinstance(d.index, pd.RangeIndex):
                        idx_as_date = pd.to_datetime(d.index, errors='coerce')
                        if idx_as_date.notna().mean() > 0.8 and 'Date' not in d.columns:
                            d.insert(0, 'Date', idx_as_date)
                            d.reset_index(drop=True, inplace=True)
                except Exception:
                    pass
                rename_map = {}
                for c in list(d.columns):
                    lc = str(c).strip().lower().replace(' ', '').replace('_','')
                    if lc in {'date','day','sessiondate','windowstart','t','timestamp'}:
                        rename_map[c] = 'Date'
                    elif lc in {'open','o'}:
                        rename_map[c] = 'Open'
                    elif lc in {'high','h'}:
                        rename_map[c] = 'High'
                    elif lc in {'low','l'}:
                        rename_map[c] = 'Low'
                    elif lc in {'close','c'}:
                        rename_map[c] = 'Close'
                    elif lc in {'adjclose','adjustedclose','adj_close'}:
                        rename_map[c] = 'Adj Close'
                    elif lc in {'volume','v','vol'}:
                        rename_map[c] = 'Volume'
                if rename_map:
                    d = d.rename(columns=rename_map)
                if 'Date' not in d.columns and len(d.columns) > 0:
                    try:
                        cand = pd.to_datetime(d.iloc[:,0], errors='coerce')
                        if cand.notna().mean() > 0.8:
                            d.insert(0, 'Date', cand)
                    except Exception:
                        pass
                if 'Close' not in d.columns and 'Adj Close' in d.columns:
                    d['Close'] = pd.to_numeric(d['Adj Close'], errors='coerce')
                for col in ['Open','High','Low','Close','Volume']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                if 'Date' in d.columns and 'Close' in d.columns:
                    d['Date'] = pd.to_datetime(d['Date'], errors='coerce', utc=True)
                    d = d.dropna(subset=['Date','Close'])
                    d['Date'] = d['Date'].dt.tz_convert(None) if hasattr(d['Date'].dt, 'tz_convert') else d['Date']
                    d = d.sort_values('Date').reset_index(drop=True)
                    return d
                return None
            best = None
            for sheet in xl.sheet_names:
                try:
                    cand = _try_sheet_to_ohlc(xl.parse(sheet))
                    if cand is not None and len(cand) >= 5:
                        best = cand
                        if str(sheet).lower() in {'daily','prices','ohlc','price','history'}:
                            break
                except Exception:
                    continue
            if best is not None:
                return best
        except Exception as e:
            raise RuntimeError(f"Failed reading uploaded Excel: {e}")

    # 1) Primary source: reports Excel (e.g., reports/AAPL_technicals.xlsx)
    data_root = _normalize_host_path(data_root) or data_root
    reports_dir = _normalize_host_path(reports_dir or os.path.join(data_root, 'reports'))
    # exact Excel override path (string path) wins if exists
    if excel_path_override:
        excel_path_override = _normalize_host_path(excel_path_override)
    path_xlsx = excel_path_override or os.path.join(reports_dir or '', f"{t}_technicals.xlsx")
    # Normalize common user input issues (extra quotes/spaces, mixed separators)
    path_xlsx = _normalize_host_path(path_xlsx) or path_xlsx
    reports_dir = _normalize_host_path(reports_dir) or reports_dir

    # If not exactly in reports_dir, try to discover anywhere under data_root
    if not os.path.exists(path_xlsx):
        try:
            import glob as _glob
            pattern = os.path.join(_normalize_host_path(data_root) or data_root, '**', f'{t}_technicals.xlsx')
            candidates = sorted(_glob.glob(pattern, recursive=True))
            if candidates:
                path_xlsx = candidates[0]
        except Exception:
            pass

    if not os.path.exists(path_xlsx) and auto_generate_report and technicals_script:
            # Try to generate the report via external script
            try:
                import sys, subprocess
                # Attempt common CLIs
                tried_cmds = [
                    [sys.executable, technicals_script, t, reports_dir],
                    [sys.executable, technicals_script, '--ticker', t, '--out', reports_dir],
                    [sys.executable, technicals_script, '--ticker', t],
                ]
                for cmd in tried_cmds:
                    try:
                        subprocess.run(cmd, check=True, timeout=180, capture_output=True)
                        break
                    except Exception:
                        continue
            except Exception:
                pass  # Non-fatal; will try reading if the script happened to succeed

    if os.path.exists(path_xlsx):
            try:
                try:
                    import openpyxl  # noqa: F401
                    xl = pd.ExcelFile(path_xlsx, engine='openpyxl')
                except ImportError as _e:
                    raise RuntimeError("Excel engine 'openpyxl' is required to read .xlsx reports. Install with: pip install openpyxl")
                def _try_sheet_to_ohlc(_df: pd.DataFrame) -> pd.DataFrame | None:
                    d = _df.copy()
                    # If index looks like dates and not a default RangeIndex, lift to a column
                    try:
                        if not isinstance(d.index, pd.RangeIndex):
                            idx_as_date = pd.to_datetime(d.index, errors='coerce')
                            if idx_as_date.notna().mean() > 0.8 and 'Date' not in d.columns:
                                d.insert(0, 'Date', idx_as_date)
                                d.reset_index(drop=True, inplace=True)
                    except Exception:
                        pass
                    # Normalize headers
                    rename_map = {}
                    for c in list(d.columns):
                        lc = str(c).strip().lower().replace(' ', '').replace('_','')
                        if lc in {'date','day','sessiondate','windowstart','t','timestamp'}:
                            rename_map[c] = 'Date'
                        elif lc in {'open','o'}:
                            rename_map[c] = 'Open'
                        elif lc in {'high','h'}:
                            rename_map[c] = 'High'
                        elif lc in {'low','l'}:
                            rename_map[c] = 'Low'
                        elif lc in {'close','c'}:
                            rename_map[c] = 'Close'
                        elif lc in {'adjclose','adjustedclose','adj_close'}:
                            rename_map[c] = 'Adj Close'
                        elif lc in {'volume','v','vol'}:
                            rename_map[c] = 'Volume'
                    if rename_map:
                        d = d.rename(columns=rename_map)
                    # If Date still missing, try first column
                    if 'Date' not in d.columns and len(d.columns) > 0:
                        try:
                            cand = pd.to_datetime(d.iloc[:,0], errors='coerce')
                            if cand.notna().mean() > 0.8:
                                d.insert(0, 'Date', cand)
                        except Exception:
                            pass
                    # If Close missing but Adj Close present
                    if 'Close' not in d.columns and 'Adj Close' in d.columns:
                        d['Close'] = pd.to_numeric(d['Adj Close'], errors='coerce')
                    # Ensure numeric types where available
                    for col in ['Open','High','Low','Close','Volume']:
                        if col in d.columns:
                            d[col] = pd.to_numeric(d[col], errors='coerce')
                    # Final sanity: need Date + Close; for gap studies we also need Open
                    if 'Date' in d.columns and 'Close' in d.columns:
                        d['Date'] = pd.to_datetime(d['Date'], errors='coerce', utc=True)
                        d = d.dropna(subset=['Date','Close'])
                        d['Date'] = d['Date'].dt.tz_convert(None) if hasattr(d['Date'].dt, 'tz_convert') else d['Date']
                        d = d.sort_values('Date').reset_index(drop=True)
                        return d
                    return None

                best = None
                for sheet in xl.sheet_names:
                    try:
                        cand = _try_sheet_to_ohlc(xl.parse(sheet))
                        if cand is not None and len(cand) >= 10:
                            best = cand
                            # Prefer sheet names that sound like daily or price
                            if str(sheet).lower() in {'daily','prices','ohlc','price','history'}:
                                break
                    except Exception:
                        continue
                if best is not None:
                    df = best
            except Exception as e:
                # Surface Excel parsing issues without exposing paths
                raise RuntimeError(f"Failed reading Excel report: {e}")

    if df is None:
        # 2) Final fallback: build from daily_aggs_v1 flat files (CSV/CSV.GZ). This avoids Excel dependency.
        daily_root = _normalize_host_path(os.path.join(data_root, 'daily_aggs_v1')) or os.path.join(data_root, 'daily_aggs_v1')
        if not os.path.exists(daily_root):
            daily_root = None
        try:
            import glob
            parts: list[pd.DataFrame] = []
            files: list[str] = []
            if daily_root:
                pattern1 = os.path.join(daily_root, '**', '*.csv')
                pattern2 = os.path.join(daily_root, '**', '*.csv.gz')
                files = sorted(glob.glob(pattern1, recursive=True)) + sorted(glob.glob(pattern2, recursive=True))
            usecols = None
            for fp in files:
                try:
                    hdr = pd.read_csv(fp, nrows=0)
                    lower = {str(c).lower(): c for c in hdr.columns}
                    tcol = next((lower[k] for k in ('ticker','symbol','t') if k in lower), None)
                    if not tcol:
                        continue
                    # Detect column names present in this file
                    ocol = next((lower[k] for k in ('open','o') if k in lower), None)
                    hcol = next((lower[k] for k in ('high','h') if k in lower), None)
                    lcol = next((lower[k] for k in ('low','l') if k in lower), None)
                    ccol = next((lower[k] for k in ('close','c') if k in lower), None)
                    vcol = next((lower[k] for k in ('volume','v') if k in lower), None)
                    dcol = next((lower[k] for k in ('date','day','window_start','timestamp','t') if k in lower), None)
                    cols = [c for c in (tcol, dcol, ocol, hcol, lcol, ccol, vcol) if c]
                    dfp = pd.read_csv(fp, usecols=cols)
                    dfp = dfp[dfp[tcol].astype(str).str.upper() == t]
                    if dfp.empty:
                        continue
                    # Normalize
                    dfp = dfp.rename(columns={
                        tcol: 'Ticker', dcol: 'Date', ocol or '': 'Open', hcol or '': 'High', lcol or '': 'Low', ccol or '': 'Close', vcol or '': 'Volume'
                    })
                    # Parse date robustly
                    rawd = dfp['Date']
                    if pd.api.types.is_numeric_dtype(rawd):
                        mx = pd.to_numeric(rawd, errors='coerce').dropna().astype(float).max() if len(rawd) else 0
                        if mx > 1e12:
                            parsed = pd.to_datetime(rawd, unit='ns', errors='coerce', utc=True)
                        elif mx > 1e9:
                            parsed = pd.to_datetime(rawd, unit='s', errors='coerce', utc=True)
                        else:
                            parsed = pd.to_datetime(rawd, errors='coerce', utc=True)
                    else:
                        parsed = pd.to_datetime(rawd, errors='coerce', utc=True)
                    dfp['Date'] = parsed.dt.tz_convert(None) if hasattr(parsed.dt, 'tz_convert') else parsed
                    for col in ['Open','High','Low','Close','Volume']:
                        if col in dfp.columns:
                            dfp[col] = pd.to_numeric(dfp[col], errors='coerce')
                    dfp = dfp.dropna(subset=['Date','Close'])
                    parts.append(dfp[['Date','Open','High','Low','Close','Volume']])
                except Exception:
                    continue
            if parts:
                df = pd.concat(parts, ignore_index=True)
        except Exception as e:
            df = None

    # 3) Final safety: Yahoo Finance fallback (optional)
    if df is None and allow_yahoo_fallback:
        try:
            import yfinance as _yf
            ydf = _yf.download(
                tickers=str(t),
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by=None,
            )
            if ydf is not None and not ydf.empty:
                if isinstance(ydf.index, pd.DatetimeIndex):
                    out = pd.DataFrame({
                        'Date': pd.to_datetime(ydf.index, utc=True),
                        'Open': pd.to_numeric(ydf.get('Open'), errors='coerce'),
                        'High': pd.to_numeric(ydf.get('High'), errors='coerce'),
                        'Low': pd.to_numeric(ydf.get('Low'), errors='coerce'),
                        'Close': pd.to_numeric(ydf.get('Close') if 'Close' in ydf.columns else ydf.get('Adj Close'), errors='coerce'),
                        'Volume': pd.to_numeric(ydf.get('Volume'), errors='coerce'),
                    })
                    out = out.dropna(subset=['Date','Close'])
                    out['Date'] = out['Date'].dt.tz_convert(None)
                    out = out.sort_values('Date').reset_index(drop=True)
                    df = out
        except Exception:
            df = None

    # If still nothing, raise a generic error (no paths)
    if df is None:
        raise ValueError("No historical data found via Excel, flat files, or Yahoo fallback.")

    # Normalize columns case-insensitively
    cols = {str(c).lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            key = n.lower()
            if key in cols:
                return cols[key]
        return None

    c_open = pick('open')
    c_high = pick('high')
    c_low = pick('low')
    c_close = pick('close')
    c_volume = pick('volume','v')
    c_date = pick('date','day','session_date','window_start','t','timestamp')

    if c_date is None or c_open is None or c_close is None:
        raise ValueError(f"Unexpected schema for {t}. Columns: {list(df.columns)}")

    # Parse date robustly (supports epoch ns/s or ISO strings)
    _raw_date = df[c_date]
    try:
        import numpy as _np
        import pandas as _pd
        if _pd.api.types.is_numeric_dtype(_raw_date):
            mx = _pd.to_numeric(_raw_date, errors='coerce').dropna().astype(float).max() if len(_raw_date) else 0
            if mx > 1e12:
                parsed_date = _pd.to_datetime(_raw_date, unit='ns', errors='coerce', utc=True)
            elif mx > 1e9:
                parsed_date = _pd.to_datetime(_raw_date, unit='s', errors='coerce', utc=True)
            else:
                parsed_date = _pd.to_datetime(_raw_date, errors='coerce', utc=True)
        else:
            parsed_date = _pd.to_datetime(_raw_date, errors='coerce', utc=True)
    except Exception:
        parsed_date = pd.to_datetime(_raw_date, errors='coerce', utc=True)

    out = pd.DataFrame({
        'Date': parsed_date,
        'Open': pd.to_numeric(df[c_open], errors='coerce'),
        'High': pd.to_numeric(df[c_high], errors='coerce') if c_high in df.columns else pd.NA,
        'Low': pd.to_numeric(df[c_low], errors='coerce') if c_low in df.columns else pd.NA,
        'Close': pd.to_numeric(df[c_close], errors='coerce'),
        'Volume': pd.to_numeric(df[c_volume], errors='coerce') if c_volume in df.columns else pd.NA,
    })
    out = out.dropna(subset=['Date'])
    # Convert to naive date (no timezone) for grouping/joins; keep time for safety
    out['Date'] = out['Date'].dt.tz_convert(None) if hasattr(out['Date'].dt, 'tz_convert') else out['Date']
    out = out.sort_values('Date').reset_index(drop=True)
    return out

def _compute_gap_drop_stats(daily: pd.DataFrame, mode: str, threshold_pct: float, direction: str | None = None) -> pd.DataFrame:
    """
    Compute event table for either:
      - mode='close_drop': prior close -> today close <= -threshold
      - mode='gap': gap % at open >= threshold (direction 'Up'/'Down')
    Returns a DataFrame with Date, Gap_% , Intraday_% , Next_Overnight_% , Next_Intraday_% , Next_Total_% .
    """
    import numpy as np
    df = daily.copy()
    df['PrevClose'] = df['Close'].shift(1)
    df['NextOpen'] = df['Open'].shift(-1)
    df['NextClose'] = df['Close'].shift(-1)
    df['Gap_%'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100.0
    df['Intraday_%'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    # Event-day close-to-close move
    df['Close_to_Close_%'] = (df['Close'] - df['PrevClose']) / df['PrevClose'] * 100.0
    df['Next_Overnight_%'] = (df['NextOpen'] - df['Close']) / df['Close'] * 100.0
    df['Next_Intraday_%'] = (df['NextClose'] - df['NextOpen']) / df['NextOpen'] * 100.0
    df['Next_Total_%'] = (df['NextClose'] - df['Close']) / df['Close'] * 100.0
    df['Next_Close_to_Close_%'] = df['Next_Total_%']

    if mode == 'close_drop':
        drop_pct = (df['Close'] - df['PrevClose']) / df['PrevClose'] * 100.0
        # Signed threshold: positive means up >= threshold; negative means down <= threshold
        if threshold_pct >= 0:
            mask = drop_pct >= threshold_pct
        else:
            mask = drop_pct <= threshold_pct
    elif mode == 'gap':
        # Signed threshold: positive means gap up >= threshold; negative means gap down <= threshold
        if threshold_pct >= 0:
            mask = df['Gap_%'] >= threshold_pct
        else:
            mask = df['Gap_%'] <= threshold_pct
    else:
        raise ValueError("mode must be 'close_drop' or 'gap'")

    if mode == 'close_drop':
        cols = ['Date','Close_to_Close_%','Next_Overnight_%','Next_Intraday_%','Next_Close_to_Close_%']
    else:
        cols = ['Date','Gap_%','Intraday_%','Next_Overnight_%','Next_Intraday_%','Next_Total_%']
    out = df.loc[mask, cols].dropna().reset_index(drop=True)
    return out

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

    try:
            # Map allowed intraday periods for Yahoo
            def best_period_for(interval_str: str, desired: str | None) -> str:
                if interval_str == "1m":
                    return "7d"  # Yahoo max for 1m
                if interval_str in {"5m", "15m", "30m", "60m", "1h"}:
                    return desired if desired in {"5d", "7d", "14d", "30d", "60d"} else "30d"
                return desired or "1y"

            # Normalize symbol for futures continuous aliases (ES -> ES=F, etc.)
            # and build candidate list for specific contracts (e.g., ESZ24)
            contract_candidates = build_futures_contract_candidates(ticker)
            if contract_candidates:
                fetch_tickers = contract_candidates
            else:
                fetch_tickers = [normalize_input_symbol(ticker)]

            if intraday:
                p = best_period_for(interval, str(period))
                df = None
                for tk in fetch_tickers:
                    df = fetch_ohlc_with_fallback(tk, interval=interval, period=p)
                    if df is not None and not df.empty:
                        break
            else:
                # Ensure valid date order and include selected end date (Yahoo end is exclusive)
                s = min(pd.to_datetime(start), pd.to_datetime(end)).date()
                e = max(pd.to_datetime(start), pd.to_datetime(end)).date()
                e_inclusive = (e + timedelta(days=1)).isoformat()
                df = None
                for tk in fetch_tickers:
                    df = fetch_ohlc_with_fallback(tk, interval=interval, start=s.isoformat(), end=e_inclusive)
                    if df is not None and not df.empty:
                        break

                # Daily-specific resilience: widen window if empty, try Ticker().history, then period fallback
                if (df is None or df.empty) and interval == "1d":
                    s_wide = (s - timedelta(days=3)).isoformat()
                    e_wide_inclusive = (e + timedelta(days=3 + 1)).isoformat()
                    if df is None or df.empty:
                        for tk in fetch_tickers:
                            df = fetch_ohlc_with_fallback(tk, interval=interval, start=s_wide, end=e_wide_inclusive)
                            if df is not None and not df.empty:
                                break
                    # Try direct Ticker().history as another path (sometimes succeeds when download() fails)
                    if (df is None or df.empty):
                        try:
                            tkr = yf.Ticker(fetch_tickers[0])
                            df_hist = tkr.history(start=s.isoformat(), end=e_inclusive, interval="1d", auto_adjust=False)
                            if isinstance(df_hist, pd.DataFrame) and not df_hist.empty:
                                df = df_hist
                        except Exception as e:
                            try:
                                st.session_state['last_fetch_errors'].append(f"yfinance Ticker.history: {e}")
                            except Exception:
                                pass
                    if df is None or df.empty:
                        span_days = (e - s).days + 1
                        period_map = [(7, "7d"), (31, "1mo"), (93, "3mo"), (183, "6mo"), (365, "1y"), (730, "2y")]
                        sel_period = "1y"
                        for lim, per in period_map:
                            if span_days <= lim:
                                sel_period = per
                                break
                        for tk in fetch_tickers:
                            df = fetch_ohlc_with_fallback(tk, interval=interval, period=sel_period)
                            if df is not None and not df.empty:
                                break
                    # Final safety: fetch full history and slice
                    if df is None or df.empty:
                        try:
                            full = None
                            for tk in fetch_tickers:
                                full = fetch_ohlc_with_fallback(tk, interval="1d", period="max")
                                if full is not None and not full.empty:
                                    break
                            if isinstance(full, pd.DataFrame) and not full.empty and isinstance(full.index, pd.DatetimeIndex):
                                df = full[(full.index.date >= s) & (full.index.date <= e)]
                        except Exception as e:
                            try:
                                st.session_state['last_fetch_errors'].append(f"max-period slice fallback: {e}")
                            except Exception:
                                pass

            if df is None or df.empty:
                details = None
                try:
                    errs = st.session_state.get('last_fetch_errors')
                    if errs:
                        details = " | Details: " + " | ".join(errs[-3:])
                except Exception:
                    pass
                msg = f"No data for {ticker} @ interval={interval}. Tried polygon, yfinance, and yahooquery."
                if details:
                    msg += details
                st.warning(msg)
                st.stop()

            df = normalize_ohlcv(df)
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Determine VWAP anchor index from input (overlay not required for backtest)
            vwap_idx = None
            if vwap_anchor:
                try:
                    anchor_dt = pd.to_datetime(vwap_anchor)
                    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                        pos = df.index.get_indexer([anchor_dt], method='nearest')
                        vwap_idx = int(pos[0]) if pos.size and pos[0] != -1 else None
                except Exception:
                    vwap_idx = None

            # Count rows for subplots (optional Volume, RSI, Stoch, MACD panel)
            base_rows = 1 + (1 if show_volume else 0)
            rows = base_rows + (1 if show_rsi else 0) + (1 if show_sto else 0) + (1 if show_macd else 0)
            row_heights = [0.55]
            if show_volume:
                row_heights.append(0.13)
            if show_rsi:
                row_heights.append(0.09)
            if show_sto:
                row_heights.append(0.09)
            if show_macd:
                row_heights.append(0.09)
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

            # Plotting block (encapsulate to align indentation)
            if True:
                if True:

                    # Main candlestick
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
                        # Place label just inside the plotting area so it's not clipped
                        fig.add_annotation(
                            xref="paper", x=0.995, y=y_val,
                            xanchor="right", yanchor="middle",
                            text=f"{trace_name}: {y_val:.2f}",
                            font=dict(color=color, size=13),
                            showarrow=False, align="right",
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

                    # Optional Volume panel
                    if show_volume:
                        if "Volume" in df.columns and df["Volume"].notna().any():
                            vol_colors = np.where(close >= df.get("Open", close), "rgba(0,200,0,0.6)", "rgba(200,0,0,0.6)")
                            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, showlegend=False), row=2, col=1)
                        else:
                            fig.add_trace(go.Bar(x=df.index, y=[0]*len(df), name="Volume", showlegend=False), row=2, col=1)

                    next_row = 2 + (1 if show_volume else 0)
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
                        next_row += 1

                    if show_macd:
                        macd_line, signal_line, hist = macd(close, int(macd_fast), int(macd_slow), int(macd_signal))
                        fig.add_trace(go.Scatter(x=df.index, y=macd_line, mode="lines", name="MACD"), row=next_row, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=signal_line, mode="lines", name="Signal"), row=next_row, col=1)
                        fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist", marker_color=np.where(hist>=0, "#2ecc40", "#ff4136")), row=next_row, col=1)
                        fig.update_yaxes(title_text="MACD", row=next_row, col=1)
                        next_row += 1

                    last_ts = pd.to_datetime(df.index[-1])
                    extend_right_edge(fig, last_ts, interval, rows)

                    # --- Run backtest if enabled and strategy is selected ---
                    trades = []
                    backtest_missing_anchor = False
                    if enable_backtest:
                        def backtest_hold_condition(price_series: pd.Series, cond: pd.Series):
                            cond = cond.fillna(False)
                            entries = (cond & ~cond.shift(1).fillna(False))
                            exits = (~cond & cond.shift(1).fillna(False))
                            in_pos = False
                            entry_idx = None
                            out = []
                            for i, ts in enumerate(price_series.index):
                                if entries.iloc[i] and not in_pos:
                                    in_pos = True
                                    entry_idx = i
                                elif exits.iloc[i] and in_pos:
                                    out.append({
                                        'entry_time': price_series.index[entry_idx],
                                        'entry_price': float(price_series.iloc[entry_idx]),
                                        'exit_time': ts,
                                        'exit_price': float(price_series.iloc[i]),
                                        'pnl': float(price_series.iloc[i] - price_series.iloc[entry_idx])
                                    })
                                    in_pos = False
                                    entry_idx = None
                            return out

                        if strategy == "Price crosses above VWAP":
                            if vwap_idx is None:
                                backtest_missing_anchor = True
                            else:
                                vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                                trades = backtest_price_crosses_vwap(close, vwap_series)
                        elif strategy == "RSI crosses above 70":
                            r = rsi(close, int(rsi_len))
                            hold = r > 70
                            trades = backtest_hold_condition(close, hold)
                        elif strategy == "RSI crosses below 30":
                            r = rsi(close, int(rsi_len))
                            hold = r < 30
                            trades = backtest_hold_condition(close, hold)
                        elif strategy == "Bollinger Bands":
                            lb, mb, ub = bbands(close, int(bb_len), float(bb_std))
                            hold = (close > ub) | (close < lb)
                            trades = backtest_hold_condition(close, hold)

                        # Plot buy/sell markers on chart
                        for t in trades:
                            fig.add_trace(go.Scatter(
                                x=[t['entry_time']], y=[t['entry_price']],
                                mode="markers", marker_symbol="triangle-up", marker_color="#00ff00", marker_size=12,
                                name="Buy"
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=[t['exit_time']], y=[t['exit_price']],
                                mode="markers", marker_symbol="triangle-down", marker_color="#ff0000", marker_size=12,
                                name="Sell"
                            ), row=1, col=1)

                    fig.update_layout(title=f"{ticker} - {interval}", xaxis_rangeslider_visible=False, height=base_height,
                                      margin=dict(l=50, r=120, t=50, b=50))
                    style_axes(fig, dark=(template == "plotly_dark"), rows=rows)

                    st.plotly_chart(fig, use_container_width=True)
                    provider = None
                    try:
                        provider = st.session_state.get('last_fetch_provider')
                    except Exception:
                        pass
                    src = f" | Source: {provider}" if provider else ""
                    st.caption(f"Rows: {len(df)} | Columns: {list(df.columns)}{src}")

                    # --- Historical Stats (from uploaded Excel only) ---
                    with st.expander("Historical Gap/Drop Stats"):
                        # Parquet-first: no Excel upload; rely on per-ticker parquet dir
                        stats_ticker = ticker
                        # Show which parquet path will be used (if found)
                        try:
                            pq_path_hint = _autofind_parquet_path(stats_ticker)
                            if pq_path_hint:
                                st.caption(f"Parquet source: {pq_path_hint}")
                            else:
                                st.caption("Parquet source: not found — set directory in sidebar.")
                        except Exception:
                            pass
                        mode = st.selectbox("Study", [
                            "Close down N% day -> next day",
                            "Gap up/down >= N% -> same day + next day",
                        ], index=0)
                        threshold = st.number_input(
                            "Threshold (%) — use + for up, - for down",
                            min_value=-50.0,
                            max_value=50.0,
                            value=-3.0,
                            step=0.1,
                        )

                        run = st.button("Run stats", key="run_gap_stats")
                        if run:
                            try:
                                # Load strictly from per-ticker Parquet (no Excel dependency)
                                daily_hist = _load_polygon_daily_for_ticker(
                                    data_root="",
                                    ticker=stats_ticker,
                                    reports_dir=None,
                                    technicals_script=None,
                                    auto_generate_report=False,
                                    excel_override=None,
                                    excel_path_override=None,
                                    allow_yahoo_fallback=False,
                                )
                                if daily_hist is None or daily_hist.empty:
                                    st.error("No historical data available from Parquet. Set 'Per-ticker Parquet directory' in the sidebar and ensure <TICKER>.parquet exists.")
                                    st.stop()
                                m = 'close_drop' if mode.startswith("Close") else 'gap'
                                thr = float(threshold)
                                result_df = _compute_gap_drop_stats(daily_hist, m, thr, None)
                                st.caption(f"Matches: {len(result_df)} events")
                                if not result_df.empty:
                                    # Summary KPIs
                                    c1,c2,c3,c4 = st.columns(4)
                                    if m == 'close_drop':
                                        with c1:
                                            st.metric("Avg Close?Close %", f"{result_df['Close_to_Close_%'].mean():.2f}%")
                                        with c2:
                                            st.metric("Avg Next Overnight %", f"{result_df['Next_Overnight_%'].mean():.2f}%")
                                        with c3:
                                            st.metric("Avg Next Intraday %", f"{result_df['Next_Intraday_%'].mean():.2f}%")
                                        with c4:
                                            st.metric("Avg Next Close?Close %", f"{result_df['Next_Close_to_Close_%'].mean():.2f}%")
                                    else:
                                        with c1:
                                            st.metric("Avg Intraday %", f"{result_df['Intraday_%'].mean():.2f}%")
                                        with c2:
                                            st.metric("Avg Next Overnight %", f"{result_df['Next_Overnight_%'].mean():.2f}%")
                                        with c3:
                                            st.metric("Avg Next Intraday %", f"{result_df['Next_Intraday_%'].mean():.2f}%")
                                        with c4:
                                            st.metric("Avg Next Total %", f"{result_df['Next_Total_%'].mean():.2f}%")

                                    st.dataframe(result_df, use_container_width=True)
                                else:
                                    st.info("No matching events for the selected criteria.")
                            except Exception as e:
                                st.error(f"Stats error: {e}")

                    # Optional: TradingView fallback expander (if the TV tab isn't visible in your setup)
                    with st.expander("TradingView (embedded)"):
                        _tv_interval_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "60m": "60", "1d": "D"}
                        tv_interval = _tv_interval_map.get(interval, "D")
                        tv_theme = "dark" if template == "plotly_dark" else "light"
                        tv_symbol = tv_symbol_for(ticker)
                        tv_cfg = {
                            "container_id": "tv_container_exp",
                            "symbol": tv_symbol,
                            "interval": tv_interval,
                            "timezone": "Etc/UTC",
                            "theme": tv_theme,
                            "style": "1",
                            "hide_side_toolbar": True,
                            "allow_symbol_change": True,
                            "studies": [],
                            "autosize": True,
                        }
                        html_code = f"""
                        <div id=\"tv_container_exp\" style=\"height:{base_height}px; width:100%\"></div>
                        <script src=\"https://s3.tradingview.com/tv.js\"></script>
                        <script type=\"text/javascript\">
                          new TradingView.widget({json.dumps(tv_cfg)});
                        </script>
                        """
                        st.components.v1.html(html_code, height=base_height+20, scrolling=False)

                    # --- Backtest results section (always shows below chart when enabled) ---
                    if enable_backtest and strategy == "Price crosses above VWAP":
                        st.subheader("Backtest Trades")
                        if backtest_missing_anchor:
                            st.info("Set Anchored VWAP and provide an anchor to run this backtest.")
                        else:
                            if trades:
                                trade_df = pd.DataFrame(trades)
                                trade_df['holding_period'] = (trade_df['exit_time'] - trade_df['entry_time']).astype(str)
                                # KPI grid
                                k1, k2, k3 = st.columns(3)
                                total_trades = len(trade_df)
                                total_pnl = float(trade_df['pnl'].sum())
                                win_rate = float((trade_df['pnl'] > 0).mean() * 100.0)
                                with k1:
                                    st.metric("Total Trades", f"{total_trades}")
                                with k2:
                                    st.metric("Total P&L", f"{total_pnl:.2f}")
                                with k3:
                                    st.metric("Win Rate", f"{win_rate:.1f}%")
                                # Trades grid
                                st.dataframe(
                                    trade_df[['entry_time','entry_price','exit_time','exit_price','pnl','holding_period']],
                                    use_container_width=True,
                                )
                            else:
                                st.caption("No signals generated for the selected range and settings.")
    except Exception as e:
        st.error(f"Error fetching or plotting data: {e}")

# ---------------- TRADINGVIEW TAB ----------------
with tab3:
    st.subheader("TradingView (embedded)")
    _tv_interval_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "60m": "60", "1d": "D"}
    tv_interval = _tv_interval_map.get(interval, "D")
    tv_theme = "dark" if template == "plotly_dark" else "light"
    tv_symbol = tv_symbol_for(ticker)
    tv_cfg = {
        "container_id": "tv_container_tab",
        "symbol": tv_symbol,
        "interval": tv_interval,
        "timezone": "Etc/UTC",
        "theme": tv_theme,
        "style": "1",
        "hide_side_toolbar": True,
        "allow_symbol_change": True,
        "studies": [],
        "autosize": True,
    }
    html_code = f"""
    <div id="tv_container_tab" style="height:{base_height}px; width:100%"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({json.dumps(tv_cfg)});
    </script>
    """
    st.components.v1.html(html_code, height=base_height+20, scrolling=False)
# --- Black-Scholes Delta function ---
def bs_delta(S, K, T, r, sigma, q, kind="call"):
    """
    Black-Scholes option delta.
    S: spot price
    K: strike price
    T: time to expiry (in years)
    r: risk-free rate
    sigma: volatility (annualized)
    q: dividend yield
    kind: "call" or "put"
    """
    import math
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if kind == "call":
        from scipy.stats import norm
        return math.exp(-q * T) * norm.cdf(d1)
    else:
        from scipy.stats import norm
        return -math.exp(-q * T) * norm.cdf(-d1)

def is_equity_symbol(sym: str) -> bool:
    s = (sym or "").strip().upper()
    if not s:
        return False
    if s.startswith('^'):
        return False
    if '=' in s or ':' in s:
        return False
    # crude: letters/digits up to 6 chars
    return s.replace('.', '').isalnum() and len(s) <= 6

@st.cache_data(show_spinner=False, ttl=300)
def fetch_expirations(ticker: str, cache_buster: str | None = None) -> list[str]:
    """Fetch option expirations: Polygon -> yfinance -> yahooquery. Records diagnostics in session."""
    st.session_state['opt_errors'] = []
    # Decide order: equities -> yahooquery ? yfinance ? polygon; others -> polygon ? yfinance ? yahooquery
    eq = is_equity_symbol(ticker)
    # Polygon first (if key present)
    try:
        api_key = None
        try:
            api_key = st.session_state.get('polygon_api_key')
            if not api_key and hasattr(st, 'secrets') and ('POLYGON_API_KEY' in st.secrets):
                api_key = st.secrets['POLYGON_API_KEY']
            if not api_key:
                import os as _os
                api_key = _os.getenv('POLYGON_API_KEY')
        except Exception:
            pass
        if api_key:
            try:
                from src.data_providers.polygon_options import fetch_polygon_expirations as _poly_exps  # type: ignore
            except Exception:
                try:
                    from polygon_options import fetch_polygon_expirations as _poly_exps  # type: ignore
                except Exception:
                    _poly_exps = None
            if _poly_exps:
                _t = ticker.strip().upper()
                if _t.startswith('^'):
                    _t = _t[1:]
                exps = _poly_exps(_t, api_key=api_key)
                if exps:
                    st.session_state['opt_last_provider'] = 'polygon'
                    return exps
                else:
                    st.session_state['opt_errors'].append('polygon: no expirations returned')
    except Exception as e:
        st.session_state['opt_errors'].append(f'polygon: {e}')
    # yfinance (order depends on eq)
    try:
        t = yf.Ticker(ticker)
        exps = t.options or []
        if exps:
            st.session_state['opt_last_provider'] = 'yfinance'
            return exps
        else:
            st.session_state['opt_errors'].append('yfinance: empty options list')
    except Exception as e:
        st.session_state['opt_errors'].append(f'yfinance: {e}')
    # yahooquery (order depends on eq)
    try:
        from yahooquery import Ticker as _YQTicker
        yq = _YQTicker(ticker)
        exps = yq.options or []
        if isinstance(exps, dict):
            exps = exps.get(ticker.upper(), []) or exps.get(ticker, [])
        if exps:
            st.session_state['opt_last_provider'] = 'yahooquery'
        else:
            st.session_state['opt_errors'].append('yahooquery: empty options list')
        return exps or []
    except Exception as e:
        st.session_state['opt_errors'].append(f'yahooquery: {e}')
        return []

@st.cache_data(show_spinner=False, ttl=300)
def fetch_chain(ticker: str, expiration: str):
    """Fetch option chain for an expiration: Polygon -> yfinance -> yahooquery. Records diagnostics."""
    import pandas as _pd
    st.session_state['opt_errors'] = st.session_state.get('opt_errors', [])
    # Polygon first
    try:
        api_key = None
        try:
            api_key = st.session_state.get('polygon_api_key')
            if not api_key and hasattr(st, 'secrets') and ('POLYGON_API_KEY' in st.secrets):
                api_key = st.secrets['POLYGON_API_KEY']
            if not api_key:
                import os as _os
                api_key = _os.getenv('POLYGON_API_KEY')
        except Exception:
            pass
        if api_key:
            try:
                from src.data_providers.polygon_options import fetch_polygon_chain as _poly_chain  # type: ignore
            except Exception:
                try:
                    from polygon_options import fetch_polygon_chain as _poly_chain  # type: ignore
                except Exception:
                    _poly_chain = None
            if _poly_chain:
                _t = ticker.strip().upper()
                if _t.startswith('^'):
                    _t = _t[1:]
                calls, puts = _poly_chain(_t, expiration, api_key=api_key)
                if isinstance(calls, _pd.DataFrame) and not calls.empty:
                    st.session_state['opt_last_provider'] = 'polygon'
                    return calls, puts
                else:
                    st.session_state['opt_errors'].append('polygon: empty chain')
    except Exception as e:
        st.session_state['opt_errors'].append(f'polygon: {e}')
    # yfinance
    try:
        t = yf.Ticker(ticker)
        oc = t.option_chain(expiration)
        calls = oc.calls.copy()
        puts = oc.puts.copy()
        calls.columns = [str(c).replace(' ', '_').lower() for c in calls.columns]
        puts.columns  = [str(c).replace(' ', '_').lower()  for c in puts.columns]
        st.session_state['opt_last_provider'] = 'yfinance'
        return calls, puts
    except Exception as e:
        st.session_state['opt_errors'].append(f'yfinance: {e}')
    # yahooquery
    try:
        from yahooquery import Ticker as _YQTicker
        yq = _YQTicker(ticker)
        df = yq.option_chain(expiration)
        if df is None or (isinstance(df, _pd.DataFrame) and df.empty):
            st.session_state['opt_errors'].append('yahooquery: empty chain')
            return _pd.DataFrame(), _pd.DataFrame()
        if not isinstance(df, _pd.DataFrame):
            try:
                df = _pd.DataFrame(df)
            except Exception as e:
                st.session_state['opt_errors'].append(f'yahooquery: {e}')
                return _pd.DataFrame(), _pd.DataFrame()
        df.columns = [str(c).replace(' ', '_').lower() for c in df.columns]
        type_col = None
        for c in ("option_type", "type", "contracttype"):
            if c in df.columns:
                type_col = c
                break
        calls = df.copy()
        puts = df.copy()
        if type_col:
            calls = df[df[type_col].astype(str).str.lower().str.startswith('c')].copy()
            puts  = df[df[type_col].astype(str).str.lower().str.startswith('p')].copy()
        elif 'contractsymbol' in df.columns:
            cs = df['contractsymbol'].astype(str)
            calls = df[cs.str.contains('C', case=False, regex=False)].copy()
            puts  = df[cs.str.contains('P', case=False, regex=False)].copy()
        st.session_state['opt_last_provider'] = 'yahooquery'
        return calls, puts
    except Exception as e:
        st.session_state['opt_errors'].append(f'yahooquery: {e}')
        return _pd.DataFrame(), _pd.DataFrame()

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
                    try:
                        diag = st.session_state.get('opt_errors') or []
                        prov = st.session_state.get('opt_last_provider', '-')
                        if diag:
                            st.caption(f"Options diagnostics (last provider={prov}): {' | '.join(diag[-3:])}")
                    except Exception:
                        pass
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
                if (calls is None or calls.empty) and (puts is None or puts.empty):
                    st.warning("No option chain returned for this expiration.")
                    try:
                        diag = st.session_state.get('opt_errors') or []
                        prov = st.session_state.get('opt_last_provider', '-')
                        if diag:
                            st.caption(f"Options diagnostics (last provider={prov}): {' | '.join(diag[-3:])}")
                    except Exception:
                        pass
                    st.stop()

            # Add delta using BS with yfinance implied volatility if available
            S = spot_price(ticker)
            exp_dt = datetime.strptime(sel, "%Y-%m-%d").date()
            T = max((exp_dt - datetime.now(timezone.utc).date()).days, 0) / 365.0

            def add_delta(df: pd.DataFrame, kind: str) -> pd.DataFrame:
                out = df.copy()
                # Resolve columns robustly
                iv_col = None
                for c in ("implied_volatility", "impliedvolatility", "iv", "impl_vol"):
                    if c in out.columns:
                        iv_col = c
                        break
                strike_col = None
                for c in ("strike", "strike_price", "strikeprice", "k"):
                    if c in out.columns:
                        strike_col = c
                        break
                if S is None or T <= 0 or iv_col is None or strike_col is None:
                    out["delta"] = np.nan
                    return out
                iv = pd.to_numeric(out[iv_col], errors="coerce").astype(float)
                strikes = pd.to_numeric(out[strike_col], errors="coerce").astype(float)
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
                moneyness = st.selectbox("Moneyness", ["All", "OTM", "ATM (ï¿½1%)", "ITM"])

            # moneyness tagging (uses spot S if available)
            def tag_filter(df: pd.DataFrame, kind: str) -> pd.DataFrame:
                out = df.copy()
                # find strike column
                s_col = None
                for c in ("strike", "strike_price", "strikeprice", "k"):
                    if c in out.columns:
                        s_col = c
                        break
                if S is not None and s_col is not None:
                    out["moneyness"] = (out[s_col] - S) / S
                    if kind == "call":
                        itm_mask = out[s_col] < S
                    else:
                        itm_mask = out[s_col] > S
                    if moneyness == "OTM":
                        out = out[~itm_mask]
                    elif moneyness == "ITM":
                        out = out[itm_mask]
                    elif moneyness == "ATM (ï¿½1%)":
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
                fig_oi.update_layout(title=f"Open Interest by Strike ï¿½ {sel}", xaxis_title="Strike", yaxis_title="Open Interest", height=380)
                st.plotly_chart(fig_oi, use_container_width=True)

            if show_vol_chart:
                fig_vol = go.Figure()
                if "volume" in cc.columns:
                    fig_vol.add_trace(go.Scatter(x=cc["strike"], y=cc["volume"], mode="lines+markers", name="Calls Volume"))
                if "volume" in pp.columns:
                    fig_vol.add_trace(go.Scatter(x=pp["strike"], y=pp["volume"], mode="lines+markers", name="Puts Volume"))
                add_vline(fig_vol, S, "Spot")
                fig_vol.update_layout(title=f"Volume by Strike ï¿½ {sel}", xaxis_title="Strike", yaxis_title="Volume", height=380)
                st.plotly_chart(fig_vol, use_container_width=True)

            if show_iv_chart:
                fig_iv = go.Figure()
                if "implied_volatility" in cc.columns:
                    fig_iv.add_trace(go.Scatter(x=cc["strike"], y=cc["implied_volatility"]*100, mode="lines+markers", name="Calls IV%"))
                if "implied_volatility" in pp.columns:
                    fig_iv.add_trace(go.Scatter(x=pp["strike"], y=pp["implied_volatility"]*100, mode="lines+markers", name="Puts IV%"))
                add_vline(fig_iv, S, "Spot")
                fig_iv.update_layout(title=f"Implied Volatility (Smile) ï¿½ {sel}", xaxis_title="Strike", yaxis_title="IV (%)", height=380)
                st.plotly_chart(fig_iv, use_container_width=True)

            # theme for options charts (match main)
            for f in ["fig_oi", "fig_vol", "fig_iv"]:
                pass  # (plotly template is fine; page-wide colors already set)

            if S is not None:
                st.caption(f"Spot ï¿½ {S:.2f} | Expirations: {len(exps)} | Showing: {sel}")
            else:
                st.caption(f"Spot unavailable | Expirations: {len(exps)} | Showing: {sel}")

        except Exception as e:
            st.error(f"Options error: {e}")



