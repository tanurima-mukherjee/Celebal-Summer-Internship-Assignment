# House Price Predictor - Streamlit ML App

A simple Streamlit application that uses linear regression to predict house prices based on house size in square feet.

## 🔗 Live Demo

[Click here to try the app on Streamlit Cloud]([(https://house-price-prediction-by-tanurima-mukherjee-assignment7.streamlit.app/)](https://house-price-prediction-by-tanurima-mukherjee-assignment7.streamlit.app/))

## Features

- **Simple Linear Regression**: Predicts house prices using only the house size
- **Interactive UI**: Adjust house size using a slider to see price predictions
- **Data Visualization**: View the dataset and regression line
- **Model Information**: See the model coefficients and performance metrics

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Access the app**:
   Open your browser and go to `http://localhost:8501`

## How to Use

1. Use the input field to enter a house size in square feet
2. Click the "Predict Price" button to see the estimated house price
3. View the calculation breakdown showing base price and size factor
4. Explore the "Data Visualization" tab to see the regression line and data points

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Required Python packages
