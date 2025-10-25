import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd


def summarize(df: pd.DataFrame) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "rows=0 first=None last=None cols=[]"
    try:
        first = str(df.index[0])
        last = str(df.index[-1])
    except Exception:
        first = last = None
    cols = list(df.columns)
    return f"rows={len(df)} first={first} last={last} cols={cols}"


def try_yf_download(tkr: str, interval: str, period: str | None, start: str | None, end: str | None):
    import yfinance as yf
    kw = dict(interval=interval, progress=False, auto_adjust=False, threads=False)
    if period:
        kw["period"] = period
    else:
        kw["start"], kw["end"] = start, end
    try:
        df = yf.download(tkr, **kw)
        print(f"[yfinance.download] OK   {summarize(df)}")
        return df, None
    except Exception as e:
        print(f"[yfinance.download] ERR  {e}")
        return None, e


def try_yf_ticker_history(tkr: str, interval: str, period: str | None, start: str | None, end: str | None):
    import yfinance as yf
    try:
        tk = yf.Ticker(tkr)
        if period:
            df = tk.history(period=period, interval=interval, auto_adjust=False)
        else:
            df = tk.history(start=start, end=end, interval=interval, auto_adjust=False)
        print(f"[yfinance.Ticker.history] OK   {summarize(df)}")
        return df, None
    except Exception as e:
        print(f"[yfinance.Ticker.history] ERR  {e}")
        return None, e


def try_yahooquery(tkr: str, interval: str, period: str | None, start: str | None, end: str | None):
    try:
        from yahooquery import Ticker
    except Exception as e:
        print(f"[yahooquery] SKIP not installed: {e}")
        return None, e
    try:
        yq = Ticker(tkr)
        if period:
            df = yq.history(period=period, interval=interval)
        else:
            df = yq.history(start=start, end=end, interval=interval)
        print(f"[yahooquery.history] OK   {summarize(df)}")
        return df, None
    except Exception as e:
        print(f"[yahooquery.history] ERR  {e}")
        return None, e


def try_stooq(tkr: str, start: str | None, end: str | None):
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        print(f"[stooq] SKIP pandas-datareader missing: {e}")
        return None, e
    try:
        sym = tkr.strip().upper()
        sym = sym + ".US" if sym.isalpha() else sym
        s = pd.to_datetime(start) if start else None
        e = pd.to_datetime(end) if end else None
        df = pdr.DataReader(sym, "stooq", start=s, end=e)
        df = df.sort_index()
        print(f"[stooq.daily] OK   {summarize(df)}")
        return df, None
    except Exception as e:
        print(f"[stooq.daily] ERR  {e}")
        return None, e


def coerce_dates(period: str | None, start: str | None, end: str | None):
    if period:
        return None, None
    if start and end:
        return start, end
    # default last 180 days
    e = datetime.utcnow().date() + timedelta(days=1)
    s = e - timedelta(days=180)
    return s.isoformat(), e.isoformat()


def run_one(tkr: str, interval: str, period: str | None, start: str | None, end: str | None):
    print(f"\n=== Testing {tkr} interval={interval} period={period} start={start} end={end} ===")
    start, end = coerce_dates(period, start, end)

    try_yf_download(tkr, interval, period, start, end)
    try_yf_ticker_history(tkr, interval, period, start, end)

    # yahooquery supports many intervals; may be throttled
    try_yahooquery(tkr, interval, period, start, end)

    # stooq only for daily
    if interval == "1d":
        try_stooq(tkr, start, end)


def main():
    ap = argparse.ArgumentParser(description="Sanity-check market data providers")
    ap.add_argument("-t", "--ticker", default="AAPL")
    ap.add_argument("-i", "--interval", default="1d", help="1d, 1h, 30m, 15m, 5m, 1m")
    ap.add_argument("-p", "--period", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--suite", choices=["none", "quick"], default="none", help="Run a small test suite")
    args = ap.parse_args()

    if args.suite == "quick":
        run_one(args.ticker, "1d", None, None, None)
        run_one(args.ticker, "5m", "5d", None, None)
        return 0

    run_one(args.ticker, args.interval, args.period, args.start, args.end)
    return 0


if __name__ == "__main__":
    sys.exit(main())

