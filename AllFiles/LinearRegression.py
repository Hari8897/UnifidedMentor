import numpy as np
import pandas as pd

d = {
    'SquareFeet': [1500, 1600, 1700, 1800, 1900],
    'Price': [300000, 320000, 340000, 360000, 380000],
    'Bedrooms': [3, 3, 4, 4, 5],
    'Age': [3, 5, 6, 10, 15]
}
df = pd.DataFrame(data=d)
X = df[['SquareFeet', 'Bedrooms', 'Age']]
y = df['Price']


# Calculate mathematical coefficients for Linear Regression
class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        # Using y = b0 + b1*x1 + b2*x2 + ... + bn*xn
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta_best[0]
        self.coefficients = theta_best[1:]
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        return X_b.dot(np.r_[self.intercept, self.coefficients])
# Linear Regression using Gradient Descent
class LinearRegressionWithGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]  # add bias term
        self.coefficients = np.zeros(X_b.shape[1])
        
        for _ in range(self.n_iterations):
            gradients = 2/m * X_b.T.dot(X_b.dot(self.coefficients) - y)
            self.coefficients -= self.learning_rate * gradients
            
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        return X_b.dot(np.r_[self.intercept, self.coefficients])
        
  
    
# Example usage
print("=========== Using Normal Equation:  =========")
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print("Predictions:", predictions)

# Output the coefficients and intercept with more clarity
print("Intercept:", model.intercept)
print("Coefficients:", model.coefficients)

print("\n")
print("=========== Using Gradient Descent:  =========")
gd_model = LinearRegressionWithGradientDescent(learning_rate=0.0000001, n_iterations=100000)
gd_model.fit(X, y)
gd_predictions = gd_model.predict(X)
print("Predictions:", gd_predictions)
# Output the coefficients and intercept with more clarity
print("Intercept:", gd_model.intercept) 
print("Coefficients:", gd_model.coefficients)
# Example usage




