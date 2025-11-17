import streamlit as st
import feedparser
from transformers import pipeline

st.title("📰 General Crypto News & Sentiment")

# Load sentiment model once
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Broad crypto news feeds
feeds = [
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
    ("Decrypt", "https://decrypt.co/feed"),
    ("CryptoSlate", "https://cryptoslate.com/feed/")
]

headlines = []
for source, url in feeds:
    feed = feedparser.parse(url)
    for entry in feed.entries[:5]:  # limit to 5 per source
        result = sentiment_model(entry.title)[0]
        headlines.append({
            "title": entry.title,
            "link": entry.link,
            "source": source,
            "sentiment": result["label"],
            "score": result["score"]
        })

# Display results
for h in headlines:
    st.write(f"**{h['title']}** ({h['source']})")
    st.write(f"Sentiment: {h['sentiment']} ({h['score']:.2f})")
    st.write(f"[Read more]({h['link']})")
    st.markdown("---")
