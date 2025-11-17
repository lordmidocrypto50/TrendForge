from fastapi import FastAPI
import requests
import feedparser
from transformers import pipeline

app = FastAPI()
sentiment_model = pipeline("sentiment-analysis")

@app.get("/price/{symbol_id}")
def get_price(symbol_id: str):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol_id}&vs_currencies=usd,cad"
    data = requests.get(url).json()
    return data.get(symbol_id, {})

@app.get("/chart/{symbol_id}")
def get_chart(symbol_id: str):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol_id}/market_chart?vs_currency=usd&days=30"
    data = requests.get(url).json()
    return data.get("prices", [])

@app.get("/news/{coin_name}")
def get_news(coin_name: str):
    feeds = [
        ("Cointelegraph", f"https://cointelegraph.com/rss/tag/{coin_name.lower()}"),
        ("Bitcoin.com", f"https://news.bitcoin.com/feed/?s={coin_name.lower()}")
    ]
    headlines = []
    for source, url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            result = sentiment_model(entry.title)[0]
            headlines.append({
                "title": entry.title,
                "link": entry.link,
                "source": source,
                "sentiment": result["label"],
                "score": result["score"]
            })
    return headlines
