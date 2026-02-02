import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Car Price Estimator", layout="wide")

st.title("How Much Is This Ford Really Worth?")
st.caption(
    "Enter a few details about the car and get a realistic price estimate based on historical Ford car data."
)

@st.cache_data
def load_data():
    return pd.read_csv("ford.csv")

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

left, center, right = st.columns([1, 2, 1])

with left:
    st.subheader("📊 Model Performance")

    st.metric(
        "R² Score",
        f"{r2:.3f}",
        help="How well the model explains car prices (closer to 1 is better).",
    )

    st.metric(
        "Average Error (MAE)",
        f"£ {int(mae):,}",
        help="On average, predictions may differ from the real price by this amount.",
    )

    st.caption(
        "Metrics are calculated on unseen test data to reflect real-world performance."
    )

with center:
    st.subheader(" Tell Me About the Car")

    year = st.slider("Manufacturing year", int(df.year.min()), int(df.year.max()), 2018)
    mileage = st.slider("How many kilometres has it run?", 0, int(df.mileage.max()), 20000)
    tax = st.slider("Annual road tax (£)", int(df.tax.min()), int(df.tax.max()), 150)
    mpg = st.slider("Fuel efficiency (MPG)", int(df.mpg.min()), int(df.mpg.max()), 50)
    engineSize = st.slider(
        "Engine size (litres)",
        float(df.engineSize.min()),
        float(df.engineSize.max()),
        1.5,
    )

    model_name = st.selectbox("Which Ford model is it?", df.model.unique())
    transmission = st.selectbox("What type of gearbox does it have?", df.transmission.unique())
    fuelType = st.selectbox("What fuel does it use?", df.fuelType.unique())

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

    currency = st.radio("Show price in:", ["GBP (£)", "INR (₹)"], horizontal=True)

    if st.button("💰 Estimate My Car’s Price", use_container_width=True):
        prediction_gbp = model.predict(input_df)[0]

        if currency == "INR (₹)":
            prediction = prediction_gbp * 100
            symbol = "₹"
        else:
            prediction = prediction_gbp
            symbol = "£"

        st.success(
            f" Based on the details you entered, this car is worth around **{symbol} {int(prediction):,}**"
        )

        st.caption(
            "This is an estimate based on historical data. Actual market prices may vary."
        )

with right:
    st.empty()
