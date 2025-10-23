import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data: X = house size (m²), y = price (£k)
X = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)
y = np.array([150, 180, 200, 230, 260, 290])

# Split into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict test data
y_pred = model.predict(X_test)

# Step 6: Show predictions
print("X_test:", X_test.flatten())
print("y_test:", y_test)
print("y_pred:", y_pred)

# Step 7: Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 8: Plot results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Size (m²)")
plt.ylabel("Price (£k)")
plt.legend()
plt.show()
