import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Synthetic generation of data examples for training the model
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    # Price in Indian Rupees (₹) - approximately 75 times USD
    price = size * 7500 + np.random.normal(0, 750000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Function for instantiating and training linear regression model
def train_model():
    df = generate_house_data()
    
    # Train-test data splitting
    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, df, mse, r2

# Main function
def main():
    st.title('🏠 House Price Predictor')
    st.write('Predict house prices based on size (square feet) using linear regression.')
    
    # Train model
    model, df, mse, r2 = train_model()
    
    # Display model information
    st.subheader("Model Information")
    st.write(f"Base Price (Intercept): ₹{model.intercept_:,.2f}")
    st.write(f"Price per Square Foot: ₹{model.coef_[0]:.2f}")
    st.write(f"Model Accuracy (R²): {r2:.4f}")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction", "Data Visualization"])
    
    with tab1:
        st.subheader("Predict House Price")
        
        # User input
        size = st.number_input('House size (square feet)', 
                              min_value=500, 
                              max_value=5000, 
                              value=1500)
        
        if st.button('Predict Price'):
            # Perform prediction
            prediction = model.predict([[size]])
            
            # Show result
            st.success(f'Estimated Price: ₹{prediction[0]:,.2f}')
            
            # Show calculation
            st.write("#### Price Calculation:")
            st.write(f"Base Price: ₹{model.intercept_:,.2f}")
            st.write(f"Size Factor: {size} sq ft × ₹{model.coef_[0]:.2f}/sq ft = ₹{size * model.coef_[0]:,.2f}")
            st.write(f"Total: ₹{model.intercept_ + size * model.coef_[0]:,.2f}")
            
            # Visualization with prediction point
            fig = px.scatter(df, x='size_sqft', y='price', 
                            title='Size vs Price Relationship',
                            labels={'size_sqft': 'House Size (sq ft)', 'price': 'Price (₹)'})
            fig.add_scatter(x=[size], y=[prediction[0]], 
                           mode='markers', 
                           marker=dict(size=15, color='red'),
                           name='Your Prediction')
            st.plotly_chart(fig)
    
    with tab2:
        st.subheader("Data Visualization")
        
        # Show sample data
        st.write("Sample Data (first 5 rows):")
        st.dataframe(df.head())
        
        # Plot regression line with Plotly
        fig = px.scatter(df, x='size_sqft', y='price', 
                        title='House Size vs Price with Regression Line',
                        labels={'size_sqft': 'House Size (sq ft)', 'price': 'Price (₹)'})
        
        # Add regression line
        x_range = np.linspace(df['size_sqft'].min(), df['size_sqft'].max(), 100)
        y_range = model.predict(pd.DataFrame(x_range, columns=['size_sqft']))
        
        fig.add_scatter(x=x_range, y=y_range, 
                       mode='lines', 
                       line=dict(color='red', width=2),
                       name='Regression Line')
        
        st.plotly_chart(fig)
        
        # Show correlation
        correlation = df.corr().iloc[0, 1]
        st.write(f"Correlation between House Size and Price: {correlation:.4f}")
        
        # Show model metrics
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² (coefficient of determination): {r2:.4f}")
        st.write("R² indicates how well the model fits the data (1.0 = perfect fit)")

if __name__ == '__main__':
    main()