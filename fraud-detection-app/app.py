import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 AI Fraud Detection System")
st.markdown("Detect fraudulent transactions using Machine Learning 🚀")

# ==============================
# FILE PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# ==============================
# DEBUG INFO (VERY IMPORTANT)
# ==============================
st.sidebar.subheader("🔍 Debug Info")

st.sidebar.write("Current Directory:")
st.sidebar.write(BASE_DIR)

st.sidebar.write("Files Available:")
try:
    st.sidebar.write(os.listdir(BASE_DIR))
except:
    st.sidebar.write("Cannot read directory")

# ==============================
# LOAD MODEL SAFELY
# ==============================
model = None
scaler = None

try:
    if not os.path.exists(model_path):
        st.error("❌ model.pkl NOT FOUND")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error("❌ scaler.pkl NOT FOUND")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

except Exception as e:
    st.error(f"❌ Error loading model files: {str(e)}")
    st.stop()

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
            st.error(f"Prediction Error: {e}")

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

            if "Class" in df.columns:
                df = df.drop("Class", axis=1)

            scaled_data = scaler.transform(df)

            predictions = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)[:, 1]

            df["Prediction"] = predictions
            df["Fraud_Probability"] = probabilities

            st.subheader("Prediction Results")
            st.write(df.head())

            # Chart
            st.subheader("Fraud Distribution")

            fraud_count = df["Prediction"].value_counts()

            plt.figure()
            plt.bar(fraud_count.index.astype(str), fraud_count.values)
            st.pyplot(plt)

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results",
                csv,
                "fraud_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Bulk Prediction Error: {e}")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.info("Built with ❤️ using Machine Learning & Streamlit")
