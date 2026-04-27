import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 AI Fraud Detection System")

menu = st.sidebar.radio("Navigation", ["Single Prediction", "Bulk Prediction"])

# ==============================
# SINGLE
# ==============================
if menu == "Single Prediction":

    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)

    features = []
    for i in range(1, 29):
        features.append(st.number_input(f"V{i}", value=0.0))

    if st.button("Predict"):

        data = np.array([[time, amount] + features])
        scaled = scaler.transform(data)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        if pred == 1:
            st.error(f"Fraud ⚠️ (Prob: {prob:.2f})")
        else:
            st.success(f"Normal ✅ (Fraud Prob: {prob:.2f})")

# ==============================
# BULK
# ==============================
else:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        scaled = scaler.transform(df)

        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1]

        df["Prediction"] = preds
        df["Fraud_Prob"] = probs

        st.write(df.head())

        # Chart
        counts = df["Prediction"].value_counts()

        plt.figure()
        plt.bar(counts.index.astype(str), counts.values)
        st.pyplot(plt)

        # Download
        csv = df.to_csv(index=False).encode()
        st.download_button("Download Results", csv, "results.csv")
