import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import feedparser
from transformers import pipeline

# --- Sentiment model ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

# --- Priority map for popular cryptos ---
PRIORITY_IDS = {
    "btc": "bitcoin", "eth": "ethereum", "xrp": "ripple", "ada": "cardano",
    "sol": "solana", "bnb": "binancecoin", "doge": "dogecoin", "ltc": "litecoin",
    "dot": "polkadot", "pol": "polygon", "dash": "dash", "etc": "ethereum_classic"
}

# --- Resolve CoinGecko ID ---
def resolve_crypto_id(user_input):
    q = user_input.strip().lower()
    if q in PRIORITY_IDS:
        return PRIORITY_IDS[q]
    coins = requests.get("https://api.coingecko.com/api/v3/coins/list").json()
    for coin in coins:
        if coin['symbol'].lower() == q:
            return coin['id']
    for coin in coins:
        if coin['id'].lower() == q or coin['name'].lower() == q:
            return coin['id']
    return None

# --- Get coin metadata (name, symbol) ---
def get_coin_metadata(symbol_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol_id}"
        data = requests.get(url, timeout=15).json()
        return {"name": data.get("name", symbol_id), "symbol": data.get("symbol", symbol_id).upper()}
    except Exception:
        return {"name": symbol_id, "symbol": symbol_id.upper()}

# --- Get price in USD/CAD ---
def get_crypto_price(symbol_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol_id}&vs_currencies=usd,cad"
    data = requests.get(url).json()
    return data.get(symbol_id)

# --- Get historical chart data ---
def get_crypto_chart(symbol_id):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol_id}/market_chart?vs_currency=usd&days=30"
    data = requests.get(url).json()
    return data.get("prices", [])

# --- Calculate RSI & MACD ---
def calculate_indicators(prices):
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("date", inplace=True)

    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    exp1 = df["price"].ewm(span=12, adjust=False).mean()
    exp2 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp1 - exp2
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df

# --- Plot chart ---
def plot_price_chart(df, label):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df.index, df["price"], label="Price (USD)")
    ax1.set_title(f"{label} - Last 30 Days")
    ax1.legend()

    ax2.plot(df.index, df["rsi"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--")
    ax2.axhline(30, color="green", linestyle="--")
    ax2.set_title("RSI (Relative Strength Index)")
    ax2.legend()

    ax3.plot(df.index, df["macd"], label="MACD", color="blue")
    ax3.plot(df.index, df["signal"], label="Signal", color="orange")
    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_title("MACD")
    ax3.legend()

    st.pyplot(fig)

# --- Interpret RSI & MACD ---
def interpret_indicators(df):
    latest_rsi = df["rsi"].dropna().iloc[-1]
    latest_macd = df["macd"].dropna().iloc[-1]
    latest_signal = df["signal"].dropna().iloc[-1]

    st.write("**Technical Analysis Summary:**")
    if latest_rsi > 70:
        st.write("- RSI indicates **overbought** conditions. Possible pullback.")
    elif latest_rsi < 30:
        st.write("- RSI indicates **oversold** conditions. Possible rebound.")
    else:
        st.write("- RSI is neutral.")

    if latest_macd > latest_signal:
        st.write("- MACD is **above** signal → Bullish momentum.")
    elif latest_macd < latest_signal:
        st.write("- MACD is **below** signal → Bearish momentum.")
    else:
        st.write("- MACD and signal are aligned → Sideways trend.")

# --- Fetch news from RSS feeds ---
def fetch_rss_news(feed_url, coin_name, ticker, source_name):
    feed = feedparser.parse(feed_url)
    query_terms = [coin_name.lower(), ticker.lower()]
    headlines = []
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        title_lower = title.lower()
        if any(term in title_lower.split()[:6] for term in query_terms) or any(f" {term} " in f" {title_lower} " for term in query_terms):
            headlines.append((title.strip(), link, source_name))
    return headlines

def fetch_crypto_news(coin_name, ticker):
    sources = [
        ("Cointelegraph", f"https://cointelegraph.com/rss/tag/{coin_name.lower()}"),
        ("Bitcoin.com", f"https://news.bitcoin.com/feed/?s={coin_name.lower()}"),
        ("Coindesk", f"https://www.coindesk.com/search?query={coin_name.lower()}")  # fallback
    ]
    headlines = []
    for source_name, url in sources:
        if "rss" in url or "feed" in url:
            headlines += fetch_rss_news(url, coin_name, ticker, source_name)
    return headlines[:5]

# --- Sentiment analysis on headlines ---
def analyze_sentiment(headlines):
    results = []
    for title, url, source in headlines:
        try:
            r = sentiment_model(title)[0]
            results.append((title, r["label"], r["score"], url, source))
        except Exception:
            results.append((title, "UNKNOWN", 0.0, url, source))
    return results

# --- UI ---
st.title("TrendForge 🔥")
st.subheader("Crypto Snapshot: Price, Chart, RSI, MACD, and Real News Sentiment")

ticker_input = st.text_input("Enter crypto ticker (e.g., BTC, ETH, XRP):")

if ticker_input:
    symbol_id = resolve_crypto_id(ticker_input)
    if not symbol_id:
        st.error("Ticker not recognized. Showing general crypto news instead:")
        headlines = fetch_crypto_news("crypto", "crypto")
        sentiments = analyze_sentiment(headlines)
        for title, label, score, url, source in sentiments:
            st.write(f"- [{title}]({url}) → {label} ({score:.2f}) — {source}")
        st.stop()

    price = get_crypto_price(symbol_id)
    chart_data = get_crypto_chart(symbol_id)

    if price:
        usd = price.get("usd")
        cad = price.get("cad")
        st.write(f"**{ticker_input.upper()}**: {usd} USD / {cad} CAD")
    else:
        st.error("Price data unavailable.")

    if chart_data:
        df = calculate_indicators(chart_data)
        plot_price_chart(df, ticker_input.upper())
        interpret_indicators(df)
    else:
        st.warning("Chart data unavailable.")

    st.write("**Latest News & Sentiment:**")
    meta = get_coin_metadata(symbol_id)
    headlines = fetch_crypto_news(meta["name"], ticker_input)
    if headlines:
        sentiments = analyze_sentiment(headlines)
        for title, label, score, url, source in sentiments:
            st.write(f"- [{title}]({url}) → {label} ({score:.2f}) — {source}")
    else:
        st.warning("No news found for this coin.")
