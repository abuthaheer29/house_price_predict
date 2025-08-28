import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")


st.title("🏠 Abu's House Price Prediction")
st.caption("Machine Learning model using Bedrooms, Bathrooms, and Sqft")
st.image(r"house.jpg", width=600) 

st.divider()  # ✅ Divider line

# -----------------------------
# Layout with 3 columns
# -----------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=10, value=3, step=1)

with col2:
    bathrooms = st.number_input("🛁 Bathrooms", min_value=1, max_value=10, value=2, step=1)

with col3:
    sqft = st.number_input("📐 Living Area (sqft)", min_value=300, max_value=10000, value=1200, step=50)

st.divider()  # ✅ Divider line
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

DATA_PATH = "house_prices.csv"

try:
    raw_df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not read {DATA_PATH}: {e}")
    st.stop()

# Keep only required columns
needed_cols = ["price", "bedrooms", "bathrooms", "sqft_living"]
df = raw_df[needed_cols].dropna().copy()
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
df = df.dropna()

# Train/Test Split & Model
X = df[["bedrooms", "bathrooms", "sqft_living"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Price"):
    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft]], 
                              columns=["bedrooms", "bathrooms", "sqft_living"])
    prediction = model.predict(input_data)[0]

    st.success(f"🏡 Estimated House Price: ₹ {prediction:,.0f}")
    st.toast("Prediction Successful!", icon="🎉")
