import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏡 House Price Prediction Dashboard")
st.write("This interactive presentation dashboard uses pandas for cleaning, matplotlib/seaborn for data analysis, and supervised machine learning algorithms for price predictions.")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('house_price_model.pkl')

model = load_model()

# Sidebar - User Inputs
st.sidebar.header("Input House Features")
area = st.sidebar.slider("Square Footage (Area)", 800, 4500, 2200)
bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3], index=1)
age = st.sidebar.slider("Age of House (Years)", 1, 40, 10)
location_score = st.sidebar.slider("Location Quality Score (1-10)", 1, 10, 7)

# Main Screen - Layout Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Price Predictor", "📊 Data Explorations & Graphs", "🤖 Algorithm Comparison"])

with tab1:
    st.subheader("Predict Price")
    
    input_data = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Age': [age],
        'Location_Score': [location_score]
    })
    
    st.write("### Selected House Parameters:")
    st.dataframe(input_data)
    
    if st.button("Calculate Predicted Price"):
        prediction = model.predict(input_data)[0]
        #st.success(f"Estimated Market Price: **${prediction:,.2f}**")
        st.success(f"Estimated Market Price: **₹{prediction:,.2f}**")

with tab2:
    st.subheader("Data Explorations & Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('graph1_price_dist.png', caption="Figure 1: Target Variable Distribution")
        st.image('graph3_correlation.png', caption="Figure 3: Correlation Matrix")
    
    with col2:
        st.image('graph2_area_vs_price.png', caption="Figure 2: Area vs Price Relationship")
        st.image('graph4_bedrooms_boxplot.png', caption="Figure 4: Bedroom Count vs Price Range")

with tab3:
    st.subheader("Supervised ML Performance")
    st.image('graph5_model_comparison.png', caption="Figure 5: Model Accuracy (R2 Score Comparison)")
    
    st.markdown("""
    ### Algorithms Used:
    1. **Linear Regression**: Fits a straight line to find relationships between features and price.
    2. **Ridge Regression**: Regularized linear model preventing overfitting.
    3. **Decision Tree Regressor**: Splits data into tree-like decision paths.
    4. **Random Forest Regressor**: Combines predictions from multiple decision trees for improved robustness.
    """)

