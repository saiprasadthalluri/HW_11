# Importing necessary libraries
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import distutils

# Loading California Housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# Spliting the dataset into training and testing sets
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, 'california_housing_model.pkl')

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Streamlit app
def main():
    st.title("California Housing Price Prediction App")
    st.write("This app predicts the median house value based on user inputs using the California Housing dataset.")

    # Load the saved model
    model = joblib.load('california_housing_model.pkl')

    # User inputs for all features
    col1, col2 = st.columns(2)
    with col1:
        MedInc = st.number_input('Median Income (in $10,000s)', min_value=0.0, max_value=20.0, value=3.0)
        HouseAge = st.number_input('House Age', min_value=0, max_value=100, value=20)
        AveRooms = st.number_input('Average Rooms per House', min_value=0.0, max_value=20.0, value=5.0)
        AveBedrms = st.number_input('Average Bedrooms per House', min_value=0.0, max_value=10.0, value=1.0)
    with col2:
        Population = st.number_input('Population in Block', min_value=0, max_value=10000, value=1000)
        AveOccup = st.number_input('Average Occupants per House', min_value=0.0, max_value=10.0, value=3.0)
        Latitude = st.number_input('Latitude', min_value=32.0, max_value=42.0, value=34.0)
        Longitude = st.number_input('Longitude', min_value=-125.0, max_value=-114.0, value=-118.0)

    # Create a dictionary to hold user input features
    input_data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }

    # Convert dictionary to DataFrame for model prediction
    input_df = pd.DataFrame([input_data])

    # Predict button
    if st.button("Predict House Value"):
        prediction = model.predict(input_df)
        st.write(f"### Predicted Median House Value: ${prediction[0] * 100000:.2f}")

    # Optional: User can provide actual price
    actual_price = st.number_input("Actual Median House Value (optional, in $100,000s)", min_value=0.0, value=0.0)
    if actual_price > 0.0:
        st.write(f"### Actual Median House Value: ${actual_price * 100000:.2f}")

    # Show evaluation metrics
    st.write("## Evaluation Metrics")
    st.write(f"- Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"- Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"- R² Score: {r2:.4f}")

if __name__ == '__main__':
    main()
