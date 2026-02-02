import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Ford Car Price Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("ford.csv")
    return df

df = load_data()

X = df.drop(columns=["price"])
y = df["price"]

X_encoded = pd.get_dummies(X, columns=["model", "transmission", "fuelType"])

scaler = StandardScaler()
num_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.33, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

col1.metric("R² Score", f"{r2:.3f}")
col2.metric("MAE", f"₹ {int(mae):,}")

st.sidebar.header("🔧 Enter Car Details")

year = st.sidebar.slider("Year", int(df.year.min()), int(df.year.max()), 2018)
mileage = st.sidebar.slider("Mileage", 0, int(df.mileage.max()), 20000)
tax = st.sidebar.slider("Tax", int(df.tax.min()), int(df.tax.max()), 150)
mpg = st.sidebar.slider("MPG", int(df.mpg.min()), int(df.mpg.max()), 50)
engineSize = st.sidebar.slider(
    "Engine Size", float(df.engineSize.min()), float(df.engineSize.max()), 1.5
)

model_name = st.sidebar.selectbox("Model", df.model.unique())
transmission = st.sidebar.selectbox("Transmission", df.transmission.unique())
fuelType = st.sidebar.selectbox("Fuel Type", df.fuelType.unique())

input_dict = {
    "year": year,
    "mileage": mileage,
    "tax": tax,
    "mpg": mpg,
    "engineSize": engineSize,
}

for col in X_encoded.columns:
    if col not in input_dict:
        input_dict[col] = 0

if f"model_{model_name}" in input_dict:
    input_dict[f"model_{model_name}"] = 1

if f"transmission_{transmission}" in input_dict:
    input_dict[f"transmission_{transmission}"] = 1

if f"fuelType_{fuelType}" in input_dict:
    input_dict[f"fuelType_{fuelType}"] = 1

input_df = pd.DataFrame([input_dict])

input_df[num_cols] = scaler.transform(input_df[num_cols])

if st.button("💰 Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Car Price: ₹ {int(prediction):,}")
