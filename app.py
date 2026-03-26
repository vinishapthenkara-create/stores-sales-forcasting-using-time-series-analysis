# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# App Title
st.title("📊 Sales Forecasting App")
st.write("Predict future sales using Machine Learning model (No pickle file)")
st.write("Sklearn installed successfully!")
# ------------------------------
# Create Dummy Training Data
# ------------------------------
np.random.seed(42)

data_size = 200

data = pd.DataFrame({
    'Year': np.random.randint(2020, 2025, data_size),
    'Month': np.random.randint(1, 13, data_size),
    'Day': np.random.randint(1, 31, data_size),
    'DayOfWeek': np.random.randint(0, 7, data_size),
    'Lag_1': np.random.randint(500, 2000, data_size),
    'Lag_2': np.random.randint(500, 2000, data_size),
    'Rolling_Mean': np.random.randint(500, 2000, data_size),
})

# Target variable (Sales)
data['Sales'] = (
    data['Lag_1'] * 0.5 +
    data['Lag_2'] * 0.3 +
    data['Rolling_Mean'] * 0.2 +
    np.random.randint(-100, 100, data_size)
)

# Features & Target
X = data.drop('Sales', axis=1)
y = data['Sales']

# ------------------------------
# Train Model
# ------------------------------
model = RandomForestRegressor()
model.fit(X, y)

# ------------------------------
# Sidebar inputs
# ------------------------------
st.sidebar.header("Enter Input Features")

year = st.sidebar.number_input("Year", 2020, 2035, 2024)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)
dayofweek = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)

lag1 = st.sidebar.number_input("Previous Day Sales (Lag_1)", value=1000.0)
lag2 = st.sidebar.number_input("2 Days Ago Sales (Lag_2)", value=950.0)
rolling_mean = st.sidebar.number_input("Rolling Mean (Last 3 Days)", value=980.0)

# Input Data
input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'DayOfWeek': [dayofweek],
    'Lag_1': [lag1],
    'Lag_2': [lag2],
    'Rolling_Mean': [rolling_mean]
})

# Show input
st.subheader("Input Data")
st.write(input_data)

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predict Sales"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales: {prediction[0]:.2f}")

# Footer
st.write("---")
st.write("Built using Streamlit | No Pickle Model")