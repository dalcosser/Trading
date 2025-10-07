import yfinance as yf
import pandas as pd

ticker = "AAPL"
df = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
print(df.head())
print("Columns:", df.columns)
