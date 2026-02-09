# Linear Regression Cheatsheet

## What is Linear Regression?

Linear regression is a supervised learning algorithm used to predict a continuous target variable based on one or more independent variables by fitting a linear equation to observed data.

### Simple Linear Regression
**One independent variable:**
```
y = β₀ + β₁x + ε
```

### Multiple Linear Regression
**Multiple independent variables:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- `y` = dependent variable (target)
- `x` = independent variable(s) (features)
- `β₀` = intercept (bias term)
- `β₁, β₂, ..., βₙ` = coefficients (weights)
- `ε` = error term (residuals)

## Matrix Form

**Prediction:**
```
ŷ = Xθ
```

**Where:**
- `X` = feature matrix (m × n), m samples, n features
- `θ` = parameter vector [β₀, β₁, ..., βₙ]ᵀ
- `ŷ` = predicted values

## Cost Function

**Mean Squared Error (MSE):**
```
J(θ) = (1/2m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²
```

Or in matrix form:
```
J(θ) = (1/2m) (Xθ - y)ᵀ(Xθ - y)
```

## Two Main Approaches to Find Optimal Parameters

### 1. Normal Equation (Closed-Form Solution)

**Formula:**
```
θ = (XᵀX)⁻¹Xᵀy
```

**Advantages:**
- ✅ No need to choose learning rate
- ✅ No iterations required
- ✅ Gives exact solution in one step
- ✅ Works well for small to medium datasets (n < 10,000)

**Disadvantages:**
- ❌ Computationally expensive for large datasets O(n³)
- ❌ Requires matrix inversion (XᵀX)⁻¹
- ❌ Doesn't work if XᵀX is singular (non-invertible)
- ❌ Slow when number of features is very large

**When to Use:** Small datasets with fewer features (<10,000)

### 2. Gradient Descent (Iterative Solution)

**Update Rule:**
```
θⱼ := θⱼ - α ∂J(θ)/∂θⱼ
```

**Gradient:**
```
∂J(θ)/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)xᵢⱼ
```

**Vectorized Update:**
```
θ := θ - (α/m) Xᵀ(Xθ - y)
```

**Types of Gradient Descent:**
1. **Batch Gradient Descent**: Uses all training samples
2. **Stochastic Gradient Descent (SGD)**: Uses one sample at a time
3. **Mini-Batch Gradient Descent**: Uses small batches of samples

**Advantages:**
- ✅ Scales well to large datasets
- ✅ Works with large number of features
- ✅ Can be used for online learning
- ✅ Memory efficient

**Disadvantages:**
- ❌ Requires choosing learning rate α
- ❌ Needs multiple iterations
- ❌ May converge to local minimum (though linear regression has only one global minimum)
- ❌ Requires feature scaling for faster convergence

**When to Use:** Large datasets with many features (>10,000)

## Key Assumptions

Linear regression relies on four critical assumptions:

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed

## Feature Scaling

Important for gradient descent convergence:

**Standardization (Z-score normalization):**
```
x' = (x - μ) / σ
```

**Min-Max normalization:**
```
x' = (x - min) / (max - min)
```

## Model Evaluation Metrics

**Mean Squared Error (MSE):**
```
MSE = (1/m) Σᵢ₌₁ᵐ (yᵢ - ŷᵢ)²
```

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE
```

**Mean Absolute Error (MAE):**
```
MAE = (1/m) Σᵢ₌₁ᵐ |yᵢ - ŷᵢ|
```

**R² Score (Coefficient of Determination):**
```
R² = 1 - (SSᵣₑₛ / SSₜₒₜ)
```
- Range: [0, 1] (can be negative for poor models)
- Closer to 1 = better fit

## Common Issues and Solutions

### Multicollinearity
**Problem:** Features are highly correlated  
**Detection:** Variance Inflation Factor (VIF) > 5-10  
**Solution:** Remove correlated features, PCA, regularization

### Overfitting
**Problem:** Model fits training data too well  
**Solution:** Regularization (Ridge/Lasso), more data, feature selection

### Underfitting
**Problem:** Model is too simple  
**Solution:** Add polynomial features, add more features, reduce regularization

## Regularization Techniques

### Ridge Regression (L2)
```
J(θ) = MSE + λΣⱼ₌₁ⁿ θⱼ²
```
- Shrinks coefficients toward zero
- Doesn't eliminate features

### Lasso Regression (L1)
```
J(θ) = MSE + λΣⱼ₌₁ⁿ |θⱼ|
```
- Can eliminate features (set to zero)
- Performs feature selection

### Elastic Net
```
J(θ) = MSE + r·λΣ|θⱼ| + ((1-r)/2)·λΣθⱼ²
```
- Combines L1 and L2 penalties

## Python Implementation Summary

**Using Normal Equation:**
```python
theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

**Using Gradient Descent:**
```python
for iteration in range(num_iterations):
    predictions = X @ theta
    errors = predictions - y
    gradient = (1/m) * X.T @ errors
    theta = theta - learning_rate * gradient
```

**Using Scikit-learn:**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Quick Decision Guide

| Scenario | Method |
|----------|--------|
| Small dataset (< 10k samples) | Normal Equation |
| Large dataset (> 10k samples) | Gradient Descent |
| Many features (> 10k) | Gradient Descent |
| Need exact solution quickly | Normal Equation |
| Online learning required | Gradient Descent |
| XᵀX is singular | Gradient Descent or Regularization |

## Time Complexity

- **Normal Equation:** O(n³) due to matrix inversion
- **Gradient Descent:** O(kmn) where k = iterations, m = samples, n = features

---

**Remember:** Linear regression is simple yet powerful. Understanding both approaches helps you choose the right tool for the problem at hand!
