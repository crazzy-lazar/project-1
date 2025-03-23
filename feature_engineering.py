import pandas as pd
import ta

def add_technical_indicators(file_path):
    """
    Adds technical indicators to the stock dataset.
    """
    # Load the dataset correctly
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Ensure "Date" is a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)

    # 🔹 Convert "Close" column to numeric (Fix)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Moving Averages
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["EMA_50"] = ta.trend.ema_indicator(df["Close"], window=50)

    # MACD (Momentum Indicator)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])

    # RSI (Overbought/Oversold Indicator)
    df["RSI"] = ta.momentum.rsi(df["Close"])

    # Bollinger Bands (Volatility Indicator)
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    # VWAP (Volume-Weighted Average Price)
    df["VWAP"] = ta.volume.volume_weighted_average_price(df["High"], df["Low"], df["Close"], df["Volume"])

    # Save processed file
    new_file_path = file_path.replace(".csv", "_features.csv")
    df.to_csv(new_file_path, index=False)

    print(f"✅ Technical Indicators Added & Saved as {new_file_path}!")
    return df

if __name__ == "__main__":
    file_path = "load_data/^NSEI_data.csv"  # Change to your file path
    df = add_technical_indicators(file_path)
    print(df.head())  # Print first few rows to verify
