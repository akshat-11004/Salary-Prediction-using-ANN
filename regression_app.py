import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from pathlib import Path

# --- PAGE SETUP ---
st.set_page_config(page_title="Salary Estimator Pro", page_icon="💸", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .result-card {
        padding: 30px;
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Setting compile=False prevents version mismatch errors
    model = tf.keras.models.load_model('regression_model.h5', compile=False)
    with open('label_encoder_gender.pkl', 'rb') as f: le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f: ohe_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: sc = pickle.load(f)
    return model, le_gender, ohe_geo, sc

try:
    model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- HEADER ---
st.title('🏦 AI Salary Estimation Dashboard')
st.markdown("Enter customer details below to predict their **Estimated Salary** using our regression model.")
st.divider()

# --- INPUT SECTION ---
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Profile")
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.radio('Gender', label_encoder_gender.classes_, horizontal=True)
        age = st.slider('Age', 18, 92, 35)

    with col2:
        st.subheader("📊 Financial Status")
        balance = st.number_input('Bank Balance ($)', min_value=0.0, value=10000.0, step=500.0)
        credit_score = st.number_input('Credit Score', value=600)
        tenure = st.slider('Tenure (Years)', 0, 10, 5)

    with col3:
        st.subheader("💳 Engagement")
        num_of_products = st.select_slider('Number of Products', options=[1, 2, 3, 4])
        has_cr_card = st.selectbox('Has Credit Card?', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox('Active Member?', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        exited = st.selectbox('Customer Exited?', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# --- PREDICTION ---
st.markdown("---")
if st.button('🚀 Calculate Estimated Salary'):
    # Prepare input data
    input_df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'Exited': [exited]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Final data assembly
    final_input = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)
    
    # Scaling
    final_input_scaled = scaler.transform(final_input)

    # Inference
    with st.spinner('Analyzing financial data...'):
        prediction = model.predict(final_input_scaled)
        salary = float(prediction[0][0])

    # Display Result
    st.markdown(f"""
        <div class="result-card">
            <h3 style="color: #6c757d;">Predicted Annual Salary</h3>
            <h1 style="color: #28a745; font-size: 50px;">${salary:,.2f}</h1>
            <p style="color: #6c757d;">Based on the provided demographic and banking profile.</p>
        </div>
    """, unsafe_allow_html=True)
