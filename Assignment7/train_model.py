import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# Generate synthetic house price data
np.random.seed(42)
n_samples = 10000

# House sizes between 1000 and 5000 square feet
house_sizes = np.random.uniform(1000, 5000, n_samples)

# House prices with some noise (base price of 100k + 100 per sq ft + noise)
house_prices = 100000 + 100 * house_sizes + np.random.normal(0, 50000, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'Size_sqft': house_sizes,
    'Price': house_prices
})

# Save dataset
data.to_csv('d:/Projects/get-github-user-details-master/Celebal-Summer-Internship-2025/Week_Assingments/Week_7/house_data.csv', index=False)

# Prepare data for model
X = data[['Size_sqft']]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model coefficients
print(f"Model Intercept: ${model.intercept_:.2f}")
print(f"Price per Square Foot: ${model.coef_[0]:.2f}")

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"R² on training data: {train_score:.4f}")
print(f"R² on test data: {test_score:.4f}")

# Save the model
model_path = 'd:/Projects/get-github-user-details-master/Celebal-Summer-Internship-2025/Week_Assingments/Week_7/house_price_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")

# Create a simple plot of the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7)
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.xlabel('House Size (square feet)')
plt.ylabel('House Price ($)')
plt.title('House Price vs Size')
plt.savefig('d:/Projects/get-github-user-details-master/Celebal-Summer-Internship-2025/Week_Assingments/Week_7/regression_plot.png')
print("Plot saved as regression_plot.png")