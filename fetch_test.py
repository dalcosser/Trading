import os
from datetime import datetime, timedelta

import yfinance as yf
from yahooquery import Ticker


# Best-effort load of .env so POLYGON_API_KEY is available from an env file
def _load_env():
    # Try python-dotenv if available; otherwise do a simple parse
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except Exception:
        pass
    # Fallback: simple .env reader (KEY=VALUE, no export, ignores comments)
    env_path = ".env"
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        # Do not overwrite existing env vars
                        if key and (key not in os.environ):
                            os.environ[key] = val
        except Exception:
            # Silently ignore parse errors; env var may already be set from shell
            pass


_load_env()


try:
    # Prefer in-repo module if available
    from src.data_providers.polygon_fetch import fetch_polygon_ohlc  # type: ignore
except Exception:
    # Allow running if src is not on sys.path or layout differs
    try:
        from polygon_fetch import fetch_polygon_ohlc  # type: ignore
    except Exception:
        fetch_polygon_ohlc = None  # type: ignore


def fetch_with_fallback(symbol="AAPL", period="5d", interval="5m"):
    print(f"\nFetching {symbol} ({interval}, {period})...")
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is not None and not df.empty:
            print(f"OK. yfinance returned {len(df)} rows.")
            return df
        else:
            print("yfinance returned no data, trying yahooquery fallback...")
    except Exception as e:
        print("yfinance error:", e)

    try:
        t = Ticker(symbol)
        df = t.history(period=period, interval=interval)
        if df is not None and not df.empty:
            print(f"OK. yahooquery returned {len(df)} rows.")
            return df
        else:
            print("yahooquery also returned no data.")
    except Exception as e:
        print("yahooquery error:", e)

    print("No usable data found.")
    return None


def fetch_today_with_polygon(symbol: str = "AAPL", interval: str = "5m", adjusted: bool = True):
    """Fetch only the current UTC day of intraday bars using Polygon.

    Reads POLYGON_API_KEY from environment (optionally loaded from .env).
    Returns a pandas DataFrame or None.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("No POLYGON_API_KEY found in env/.env; skipping Polygon fetch.")
        return None
    if fetch_polygon_ohlc is None:
        print("Polygon fetch module not available in this environment.")
        return None

    # Compute start/end as [today, tomorrow) in UTC so we only get today's data
    start = datetime.utcnow().date().isoformat()
    end = (datetime.utcnow().date() + timedelta(days=1)).isoformat()

    try:
        print(f"\nFetching today's data from Polygon for {symbol} ({interval})...")
        df = fetch_polygon_ohlc(
            symbol,
            interval=interval,
            start=start,
            end=end,
            adjusted=adjusted,
            api_key=api_key,
        )
        if df is not None and hasattr(df, "empty") and not df.empty:
            print(f"✔ Polygon returned {len(df)} rows for today (UTC).")
            return df
        print("Polygon returned no rows for today's window.")
        return None
    except Exception as e:
        print("Polygon error:", e)
        return None


if __name__ == "__main__":
    # First, try Polygon for today's intraday if key present
    pdf = fetch_today_with_polygon("AAPL", interval="5m")
    if pdf is not None:
        print(pdf.head())
    else:
        # Fall back to yfinance/yahooquery historical window
        df = fetch_with_fallback("AAPL", period="5d", interval="5m")
        if df is not None:
            print(df.head())

