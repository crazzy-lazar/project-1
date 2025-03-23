import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline

# Initialize FinBERT sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_yahoo_finance_headlines(stock="NIFTY"):
    """
    Scrapes Yahoo Finance for the latest news headlines related to the stock.
    """
    url = f"https://finance.yahoo.com/quote/{stock}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("❌ Failed to fetch news headlines.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    headlines = []

    for item in soup.find_all("h3"):
        headline = item.text.strip()
        if headline:  # Ensure it's not empty
            headlines.append(headline)

    print(f"✅ Fetched {len(headlines)} news headlines for {stock}")
    return headlines

def analyze_sentiment(headlines):
    """
    Runs FinBERT sentiment analysis on the extracted headlines.
    """
    results = sentiment_pipeline(headlines)
    sentiments = [res["label"] for res in results]
    return sentiments

def save_sentiment_data(stock="NIFTY"):
    """
    Scrapes headlines, analyzes sentiment, and saves data as CSV.
    """
    headlines = get_yahoo_finance_headlines(stock)
    if not headlines:
        print("❌ No headlines found.")
        return

    sentiments = analyze_sentiment(headlines)
    df = pd.DataFrame({"Headline": headlines, "Sentiment": sentiments})
    
    # Save to CSV
    filename = f"{stock}_sentiment.csv"
    df.to_csv(filename, index=False)
    
    print(f"✅ Sentiment data saved as {filename}!")

if __name__ == "__main__":
    save_sentiment_data("^NSEI")  # Run for NIFTY 50
