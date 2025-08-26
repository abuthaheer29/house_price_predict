import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

st.set_page_config(page_title="₹ House Price (Bedrooms + Bathrooms + Sqft)", layout="centered")
st.title("🏠 House Price Predictor")
st.caption("Features used: Bedrooms, Bathrooms, Living Area (sqft)")

# ---- Settings ----
INR_PER_USD_DEFAULT = 83.0  # Default conversion rate

# ---- Load Data ----
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

DATA_PATH = "/Users/abuthaheerm/Downloads/house_prices.csv"

try:
    raw_df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not read {DATA_PATH}: {e}")
    st.stop()

# Keep only the 3 features we want + target
needed_cols = ["price", "bedrooms", "bathrooms", "sqft_living"]
missing = [c for c in needed_cols if c not in raw_df.columns]
if missing:
    st.error(f"Missing columns in CSV: {missing}")
    st.stop()

df = raw_df[needed_cols].dropna().copy()

# Ensure bathrooms is numeric
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
df = df.dropna()

# ---- Train/Test split & Model ----
X = df[["bedrooms", "bathrooms", "sqft_living"]]
y = df["price"]  # assumed USD inside dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300, random_state=42, n_jobs=-1, max_depth=None
)
model.fit(X_train, y_train)
test_mae_usd = mean_absolute_error(y_test, model.predict(X_test))

# ---- INR Conversion Control ----
with st.sidebar:
    st.header("⚙️ Settings")
    inr_rate = st.number_input(
        "USD → INR rate",
        min_value=50.0, max_value=200.0, value=float(INR_PER_USD_DEFAULT), step=0.5,
        help="Used only to display prices in INR"
    )
    st.caption("Tip: Change the rate if you want a different INR conversion.")

# ---- Helper for quantiles ----
def q(col, low=0.01, high=0.99):
    return float(df[col].quantile(low)), float(df[col].quantile(high))

# ---- Slider ranges ----
b_lo, b_hi = int(q("bedrooms")[0]), int(q("bedrooms")[1])
ba_lo, ba_hi = q("bathrooms")   # keep float (can be 1.5, 2.5, etc.)
s_lo, s_hi = int(q("sqft_living")[0]), int(q("sqft_living")[1])

# ---- Defaults (medians) ----
b_med = int(df["bedrooms"].median())
ba_med = float(df["bathrooms"].median())
s_med = int(df["sqft_living"].median())

# ---- Input Sliders ----
st.subheader("Enter Home Details")

bedrooms = st.slider("Bedrooms", max(1, b_lo), max(b_lo + 1, b_hi), b_med)

# bathrooms → allow 0.5 steps (since data can be 1.5, 2.5 etc.)
bathrooms = st.slider(
    "Bathrooms",
    float(max(1, np.floor(ba_lo * 2) / 2)),
    float(np.ceil(ba_hi * 2) / 2),
    float(round(ba_med * 2) / 2),
    step=0.5
)

sqft_living = st.slider(
    "Living Area (sqft)", s_lo, max(s_lo + 10, s_hi), s_med, step=10
)

# ---- Predict ----
X_new = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living
}])

if st.button("🔮 Predict Price"):
    pred_price_usd = float(model.predict(X_new)[0])
    pred_price_inr = pred_price_usd * inr_rate
    test_mae_inr = test_mae_usd * inr_rate

    st.markdown("### 💰 Predicted Price")
    st.success(f"**₹ {pred_price_inr:,.0f}**")
    
    st.toast("✨ Prediction Successful!", icon="🎉")
    st.markdown("##### Model quality (for reference)")
    st.write(f"- Test MAE ≈ **₹ {test_mae_inr:,.0f}**")
    st.caption("MAE = average absolute error on a hold-out test set. Lower is better.")

    with st.expander("See the exact input sent to model"):
        st.write(X_new)
