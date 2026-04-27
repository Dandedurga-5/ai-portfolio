import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

# ==============================
# SAFE FILE PATHS (IMPORTANT FIX)
# ==============================
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Load model
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 AI Fraud Detection System")
st.markdown("Detect fraudulent transactions using Machine Learning 🚀")

# ==============================
# SIDEBAR MENU
# ==============================
menu = st.sidebar.radio("Navigation", ["🔍 Single Prediction", "📂 Bulk Prediction"])

# ==============================
# SINGLE PREDICTION
# ==============================
if menu == "🔍 Single Prediction":

    st.header("Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input("⏱️ Time", value=0.0)
    with col2:
        amount = st.number_input("💰 Amount", value=0.0)

    st.subheader("Features (V1 - V28)")

    cols = st.columns(4)
    features = []

    for i in range(28):
        with cols[i % 4]:
            val = st.number_input(f"V{i+1}", value=0.0)
            features.append(val)

    if st.button("🔍 Predict"):

        try:
            input_data = np.array([[time, amount] + features])
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            st.subheader("Result")

            if prediction == 1:
                st.error(f"⚠️ Fraud Detected (Probability: {prob:.2f})")
            else:
                st.success(f"✅ Normal Transaction (Fraud Prob: {prob:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")

# ==============================
# BULK PREDICTION
# ==============================
elif menu == "📂 Bulk Prediction":

    st.header("Upload CSV File")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Preview Data")
            st.write(df.head())

            # Drop Class column if present
            if "Class" in df.columns:
                df = df.drop("Class", axis=1)

            # Scale
            scaled_data = scaler.transform(df)

            # Predict
            predictions = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[:, 1]

            df["Prediction"] = predictions
            df["Fraud_Probability"] = probabilities

            st.subheader("Prediction Results")
            st.write(df.head())

            # ==============================
            # VISUALIZATION
            # ==============================
            st.subheader("Fraud Distribution")

            fraud_count = df["Prediction"].value_counts()

            plt.figure()
            plt.bar(fraud_count.index.astype(str), fraud_count.values)
            plt.title("Fraud vs Normal Transactions")
            plt.xlabel("Class (0 = Normal, 1 = Fraud)")
            plt.ylabel("Count")

            st.pyplot(plt)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.info("Built with ❤️ using Machine Learning & Streamlit")
