
import streamlit as st
import yfinance as yf
import pandas as pd
import joblib

st.set_page_config(page_title="Stock Trend Confidence Predictor")

st.title("📈 Stock Trend Confidence Predictor")

# Load model
model = joblib.load("/content/stock-trend-predictor/model/model.pkl")


symbol = st.text_input("Enter Stock Symbol (e.g. AAPL)", "AAPL")

if st.button("Predict"):
    data = yf.download(symbol, period="6mo")

    # Fix multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Feature engineering
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Return"].rolling(5).std()
    data["MA_ratio"] = data["MA_5"] / data["MA_10"]
    data["Price_MA5"] = data["Close"] / data["MA_5"]

    data.dropna(inplace=True)

    features = [
        "MA_5", "MA_10", "MA_ratio",
        "Price_MA5", "Return", "Volatility"
    ]

    latest = data[features].iloc[-1:].values

    prob = model.predict_proba(latest)[0]

    down_prob = prob[0] * 100
    up_prob = prob[1] * 100

    st.subheader("Prediction Confidence")
    st.metric("📈 Up Probability", f"{up_prob:.2f}%")
    st.metric("📉 Down Probability", f"{down_prob:.2f}%")

    # Confidence label
    confidence = max(up_prob, down_prob)

    if confidence > 65:
        st.success("🟢 High Confidence Signal")
    elif confidence > 55:
        st.warning("🟡 Medium Confidence Signal")
    else:
        st.error("🔴 Low Confidence – Avoid Trade")
