# ############################################
# This script showcases differences between python libraries numpy scipy and pandas
# Scenario: You have *monthly sales data* for a store. Some months are missing.
# Taks: 
# Fill missing sales values (interpolation)
# Calculate yearly sales total (sum)
# Fit a simple curve (like a trend line) to predict next month's sales (prediction)
# ############################################

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Step1 - Mock data source with missing values
data = {
    'Month': pd.date_range(start='2024-01-01', periods=12, freq='ME'), # Notice the month is ended with last day of the month, and leap year is recognized
    'Sales': [200, 220, np.nan, 250, 270, np.nan, 310, 330, 340, np.nan, 400, 420]
}

df = pd.DataFrame(data) # Pack into pandas dataframe
# print('Original data:')
# print(data)
print('\nDataFramed original data:')
print(df)

# Step2 - fill missing sales with linear interpolation (Pandas)
df['Sales'] = df['Sales'].interpolate(method='linear')
print("\nPost interpolation, data becomes")
print(df)

# Step3 - Calculate total annual sale
total_sales = np.sum(df['Sales'].values)    # .values extract values of each sale, put in numpy array
print(f"\nTotal annual sale is {total_sales:.2f}")

# Step4 - Fit a simple linear model (Scipy)
# Define a model function y = a * x + b
def linear_model(x, a, b):
    return a * x + b

# Prepare x as month number (1,2,3...12), y as sale
x_data = np.arange(len(df))
y_data = df['Sales'].values
params, covariance = curve_fit(linear_model, x_data, y_data)
a, b = params   # Keep these set of parameters, will need to plug in linear model function
print(f"\nFitted model is: Sales = {a:.2f} * Month + {b:.2f}")

# Step5 - Predict (future) next month's sale
next_month = len(df)
predicted_sale = linear_model(next_month, a, b)
print(f"Predicted next month's sale: {predicted_sale:.3f}")

# Step6 - Plot 
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], 'bo-', label='Actual Sales')
plt.plot(pd.date_range(start='2024-01-01', periods=13, freq='ME'), 
         linear_model(np.arange(13), a, b), 'r--', label='Fitted trend')
plt.title('Monthly Sale and Trend')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
