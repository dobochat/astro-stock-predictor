import swisseph as swe
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests

# Get real-time Bitcoin price (CoinGecko API)
def get_btc_price():
    try:
        res = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=inr")
        return res.json()['bitcoin']['inr']
    except:
        return "N/A"

# Function to get planetary longitudes on a date
def get_planetary_positions(date_str):
    jd = swe.julday(*map(int, date_str.split("-")))
    planets = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']
    positions = {}
    for i, p in enumerate(range(swe.SUN, swe.SATURN + 1)):
        lon, _ = swe.calc_ut(jd, p)
        positions[planets[i]] = lon
    return positions

# Load historical stock data
def download_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df[['Open', 'Close']]
    df['Date'] = df.index.date.astype(str)
    return df

# Streamlit App UI
def main():
    st.set_page_config(page_title="Astro Stock Predictor", layout="centered")
    st.title("🪐 Astro-Based Stock Market Predictor")

    st.metric("🔴 Bitcoin Price (INR)", get_btc_price())

    date = st.date_input("Select a Date", datetime.today())
    symbol = st.text_input("Stock Symbol (Yahoo Finance)", value="^NSEI")
    planet = st.selectbox("Visualize Planet vs Market", ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn'])

    if st.button("Predict Trend"):
        pos = get_planetary_positions(date.strftime('%Y-%m-%d'))
        input_df = pd.DataFrame([pos])
        try:
            model = joblib.load('astro_model.pkl')
            prediction = model.predict(input_df)[0]
            st.success(f"📈 Predicted Market Trend on {date.strftime('%Y-%m-%d')}: {prediction}")
            st.info(predict_dasha_outlook(date.strftime('%Y-%m-%d')))
        except:
            st.error("Model file not found. Please train it first.")

    if st.button("Show Planetary Chart"):
        fig, ax = plt.subplots(figsize=(10, 5))
        pos = get_planetary_positions(date.strftime('%Y-%m-%d'))
        sns.barplot(x=list(pos.keys()), y=list(pos.values()), palette='mako', ax=ax)
        ax.set_title(f'Planetary Positions on {date.strftime("%Y-%m-%d")}')
        ax.set_ylabel('Longitude (°)')
        st.pyplot(fig)

    if st.button("Plot Planet vs Market"):
        df = download_stock_data(symbol, date - timedelta(days=30), date)
        df['Date'] = pd.to_datetime(df['Date'])
        df[planet] = df['Date'].dt.strftime('%Y-%m-%d').apply(lambda d: get_planetary_positions(d)[planet])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['Date'], df['Close'], label='Market Close')
        ax.plot(df['Date'], df[planet], label=f'{planet} Degree', linestyle='--')
        ax.set_title(f'{symbol} Close vs {planet} Position')
        ax.set_xlabel('Date')
        ax.legend()
        st.pyplot(fig)

# Dasha forecast
def predict_dasha_outlook(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    if year % 7 == 0:
        return "📉 High volatility expected due to Saturn period."
    elif year % 5 == 0:
        return "💰 Potential wealth growth — Jupiter dominant year."
    else:
        return "📊 Neutral dasha period. Watch for Moon and Mars transits."

if __name__ == "__main__":
    main()
