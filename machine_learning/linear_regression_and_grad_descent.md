# Linear Regression Cheatsheet

## What is Linear Regression?

Linear regression is a supervised learning algorithm used to predict a continuous target variable based on one or more independent variables by fitting a linear equation to observed data.

### Equation

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \varepsilon$$

Where:
- $y$ = dependent variable (target)
- $x$ = independent variable(s) (features)
- $\beta_0$ = intercept (bias term)
- $\beta_1, \beta_2, \ldots, \beta_n$ = coefficients (weights)
- $\varepsilon$ = error term (residuals)

### Visual Intuition

![Linear Regression Best Fit Line](https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Normdist_regression.png/325px-Normdist_regression.png)

*The goal is to find the line that minimizes the total distance between data points and the line.*

## Matrix Form

**Prediction:**

$$\hat{y} = X\theta$$

**Where:**
- $X$ = feature matrix $(m \times n)$, m samples, n features
- $\theta$ = parameter vector $[\beta_0, \beta_1, \ldots, \beta_n]^T$
- $\hat{y}$ = predicted values

## Cost Function

**Mean Squared Error (MSE):**

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

Or in matrix form:

$$J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$$

The cost function measures how wrong our predictions are. It computes the average of the squared differences between predicted values and actual values. The factor of 2 in the denominator is a convenience that simplifies the derivative during optimization.

![Convex Cost Function](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Error_surface_of_a_linear_neuron_with_two_input_weights.png/400px-Error_surface_of_a_linear_neuron_with_two_input_weights.png)

*The MSE cost function for linear regression is CONVEX (bowl-shaped). It has exactly ONE global minimum — no local minima to worry about.*

---

## Two Main Approaches to Find Optimal Parameters

### 1. Normal Equation (Closed-Form Solution)

**Formula:**

$$\theta = (X^T X)^{-1} X^T y$$

**Advantages:**
- ✅ No need to choose learning rate
- ✅ No iterations required
- ✅ Gives exact solution in one step
- ✅ Works well for small to medium datasets (n < 10,000)

**Disadvantages:**
- ❌ Computationally expensive for large datasets $O(n^3)$ (Order n cubed)
- ❌ Requires matrix inversion $(X^T X)^{-1}$
- ❌ Doesn't work if $X^T X$ is singular (non-invertible)
- ❌ Slow when number of features is very large

**When to Use:** Small datasets with fewer features (<10,000)

---

### 2. Gradient Descent (Iterative Solution)

#### What is Gradient Descent?

Gradient Descent is an iterative optimization algorithm used to find the parameters (weights) that minimize the cost function.

Mathematically, the gradient ($\nabla J$) is a vector of partial derivatives that points in the direction of steepest **ascent**. By subtracting the gradient, we move in the direction of steepest **descent**, toward the minimum.

#### How It Works Step-by-Step

```
Step 1: Initialize parameters θ randomly or to zeros
Step 2: Compute predictions ŷ = Xθ
Step 3: Compute the error (ŷ - y)
Step 4: Compute the gradient ∇J(θ) = (1/m) Xᵀ(ŷ - y)
Step 5: Update parameters θ := θ - α · ∇J(θ)
Step 6: Repeat steps 2-5 until convergence
```

![Gradient Descent Convergence](https://upload.wikimedia.org/wikipedia/commons/a/a3/Gradient_descent.gif)

*Gradient descent with different initial conditions, iteratively stepping toward the minimum. Each step moves $\theta$ in the direction of steepest descent, with step size controlled by the learning rate ($\alpha$) and the gradient magnitude.*

#### Update Rule

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

#### Gradient

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \, x_{ij}$$

#### Vectorized Update

$$\theta := \theta - \frac{\alpha}{m} X^T (X\theta - y)$$

#### The Learning Rate ($\alpha$)

The learning rate controls the size of each step. Choosing it correctly is critical.

![Learning Rate Effects](https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/400px-Gradient_descent.svg.png)

| Learning Rate | Effect |
|:---:|:---:|
| **Too small** (0.0001) | Very slow convergence, takes thousands of iterations |
| **Just right** (0.01) | Fast, stable convergence to the minimum |
| **Too large** (10) | Overshoots the minimum, oscillates or diverges |

#### How to Choose the Learning Rate

There is no universal value — it depends on the problem, the data, and whether features are scaled.

**Step 1 — Start with a reasonable default:**
A good starting point is $\alpha = 0.01$. This works well for most problems when features are scaled.

**Step 2 — Try values on a logarithmic scale:**
Test a range of values that increase by roughly 3x each step:

$$\alpha \in \{0.001, \; 0.003, \; 0.01, \; 0.03, \; 0.1, \; 0.3, \; 1.0\}$$

**Step 3 — Plot the cost vs. iterations for each value:**
The cost curve tells you everything:

| What You See | What It Means | Action |
|:---|:---|:---|
| Cost decreases very slowly | $\alpha$ is too small | Increase $\alpha$ |
| Cost decreases smoothly and flattens | $\alpha$ is good | Keep it |
| Cost oscillates (up and down) | $\alpha$ is slightly too large | Decrease $\alpha$ by half |
| Cost explodes (increases rapidly, NaN) | $\alpha$ is way too large | Decrease $\alpha$ by 10x |

```python
# Practical example: testing multiple learning rates
learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

for lr in learning_rates:
    theta = np.zeros(n_features)
    costs = []
    for i in range(500):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta = theta - lr * gradient
        costs.append(compute_cost(X, y, theta))
    
    plt.plot(costs, label=f'α = {lr}')

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.title('Cost vs Iterations for Different Learning Rates')
plt.show()
```

**Step 4 — Pick the largest $\alpha$ that converges smoothly:**
You want fast convergence without instability. The best $\alpha$ is the largest value where the cost curve still decreases smoothly.

#### Managing the Learning Rate During Training

A fixed learning rate is often good enough for linear regression, but for more complex problems or large datasets, you may want to **decay** the learning rate over time. The idea is: take large steps early (fast progress), then smaller steps later (precision near the minimum).

**1. Step Decay — Reduce by a factor every N epochs:**

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / N \rfloor}$$

Example: Start at $\alpha_0 = 0.1$, multiply by $\gamma = 0.5$ every 100 epochs.

```python
# Step decay
def step_decay(epoch, initial_lr=0.1, drop_factor=0.5, drop_every=100):
    return initial_lr * (drop_factor ** (epoch // drop_every))
```

**2. Exponential Decay — Smooth continuous reduction:**

$$\alpha_t = \alpha_0 \cdot e^{-kt}$$

Where $k$ controls how fast the rate decays.

```python
# Exponential decay
def exp_decay(epoch, initial_lr=0.1, decay_rate=0.01):
    return initial_lr * np.exp(-decay_rate * epoch)
```

**3. Inverse Time Decay — Slow, gradual reduction:**

$$\alpha_t = \frac{\alpha_0}{1 + k \cdot t}$$

```python
# Inverse time decay
def time_decay(epoch, initial_lr=0.1, decay_rate=0.01):
    return initial_lr / (1 + decay_rate * epoch)
```

**4. Adaptive Methods (most practical) — Let the algorithm decide:**

Instead of manually tuning schedules, adaptive optimizers adjust the learning rate automatically per parameter:

| Method | Idea | When to Use |
|:---|:---|:---|
| **AdaGrad** | Scales $\alpha$ down for frequently updated parameters | Sparse data |
| **RMSProp** | Uses exponential moving average of squared gradients | Non-stationary problems |
| **Adam** | Combines momentum + RMSProp; adapts per-parameter | Default choice for most problems |

```python
# In scikit-learn, SGDRegressor supports adaptive learning rates
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    learning_rate='adaptive',  # reduces lr when loss stops improving
    eta0=0.01,                 # initial learning rate
    max_iter=1000
)
model.fit(X_train, y_train)
```

#### Quick Rules of Thumb

1. **Always scale features first** — this makes the choice of $\alpha$ much less sensitive
2. **Start with $\alpha = 0.01$** and adjust from there
3. **If cost increases** → $\alpha$ is too large, divide by 10
4. **If cost barely moves** → $\alpha$ is too small, multiply by 3
5. **Plot cost vs. iterations** — never skip this, it's your diagnostic tool
6. **For linear regression specifically**, a fixed learning rate with scaled features is almost always sufficient
7. **For production / complex models**, use Adam optimizer and let it handle the learning rate

#### Worked Example: Gradient Descent for Linear Regression

Suppose we have a tiny dataset and want to fit $y = \theta_0 + \theta_1 x$:

```
Data: x = [1, 2, 3], y = [2, 4, 5]
Initialize: θ₀ = 0, θ₁ = 0, α = 0.1

── Iteration 1 ──
Predictions: ŷ = [0+0·1, 0+0·2, 0+0·3] = [0, 0, 0]
Errors:      ŷ - y = [0-2, 0-4, 0-5] = [-2, -4, -5]
Gradient θ₀: (1/3)·(-2 + -4 + -5)           = -3.667
Gradient θ₁: (1/3)·(-2·1 + -4·2 + -5·3)     = -8.333
Update:      θ₀ = 0 - 0.1·(-3.667)   = 0.367
             θ₁ = 0 - 0.1·(-8.333)   = 0.833

── Iteration 2 ──
Predictions: ŷ = [0.367+0.833·1, 0.367+0.833·2, 0.367+0.833·3] = [1.200, 2.033, 2.867]
Errors:      ŷ - y = [-0.800, -1.967, -2.133]
Gradient θ₀: (1/3)·(-0.800 + -1.967 + -2.133)         = -1.633
Gradient θ₁: (1/3)·(-0.800·1 + -1.967·2 + -2.133·3)   = -3.711
Update:      θ₀ = 0.367 - 0.1·(-1.633)  = 0.530
             θ₁ = 0.833 - 0.1·(-3.711)  = 1.204

...after many iterations → θ₀ ≈ 0.333, θ₁ ≈ 1.500 (best-fit line)
```

#### Types of Gradient Descent

| Type | Samples per Update | Speed | Stability | Memory | Best For |
|------|--------------------|-------|-----------|--------|----------|
| **Batch GD** | All m samples | Slow per epoch | Smooth, stable | High | Small datasets |
| **Stochastic GD** | 1 random sample | Fast per update | Noisy, zigzag | Low | Online learning |
| **Mini-Batch GD** | Small batch (32-256) | Medium | Balanced | Medium | Most common in practice |

![Gradient Descent with Momentum - Convergence Paths](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Gradient_descent_with_momentum.svg/450px-Gradient_descent_with_momentum.svg.png)

*Comparison of convergence paths. Batch GD follows a smooth path, SGD zigzags noisily, and Mini-Batch GD balances both.*

**Advantages:**
- ✅ Scales well to large datasets
- ✅ Works with large number of features
- ✅ Can be used for online learning
- ✅ Memory efficient

**Disadvantages:**
- ❌ Requires choosing learning rate $\alpha$
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

> **Gradient descent requires feature scaling for fast convergence.** When features are on different scales, the cost contours become elongated and gradient descent zigzags instead of converging directly.
>
> **Always scale your features before using gradient descent.** Standardization (Z-score) is the most common choice.
>
> See **[feature_scaling.md](feature_scaling.md)** for the full guide on Standardization, Min-Max Normalization, and Robust Scaling — including when to use each, advantages/disadvantages, and Python code.

## Model Evaluation Metrics

Evaluation metrics quantify how well the model's predictions match the actual values. Different metrics emphasize different aspects of error, so choosing the right one depends on the problem.

---

### 1. Mean Squared Error (MSE)

**Definition:** The average of the squared differences between predicted and actual values. Squaring penalizes large errors more heavily than small ones.

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**Interpretation:**
- Units: squared units of the target variable (e.g., dollars² for price prediction)
- Lower = better. MSE = 0 means perfect predictions.
- A large MSE means the model makes big errors on some samples.

**When to use:**
- Optimization target for training (differentiable, convex)
- When large errors are especially undesirable (e.g., predicting structural loads)

**Advantages:**
- ✅ Mathematically convenient (smooth, differentiable)
- ✅ Penalizes large errors heavily → encourages predictions close to all points
- ✅ Standard cost function for linear regression

**Disadvantages:**
- ❌ Not in the same units as the target (hard to interpret directly)
- ❌ Very sensitive to outliers (one bad prediction dominates the metric)
- ❌ Scale-dependent (cannot compare across different datasets)

---

### 2. Root Mean Squared Error (RMSE)

**Definition:** The square root of MSE. Brings the error back to the same units as the target variable.

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2}$$

**Interpretation:**
- Units: same as target variable (e.g., dollars, degrees)
- "On average, our predictions are off by approximately RMSE units"
- Example: RMSE = 15,000 on house prices means average error ≈ $15,000

**When to use:**
- Most common reporting metric for regression
- When you want an interpretable measure of average error magnitude
- When large errors should be penalized (inherits from MSE)

**Advantages:**
- ✅ Same units as target → easily interpretable
- ✅ Penalizes large errors (inherited from squaring)
- ✅ Most widely used regression metric

**Disadvantages:**
- ❌ Still sensitive to outliers
- ❌ Scale-dependent (can't compare RMSE across different targets)
- ❌ Not directly differentiable (use MSE for optimization)

---

### 3. Mean Absolute Error (MAE)

**Definition:** The average of the absolute differences between predicted and actual values. All errors are weighted equally regardless of magnitude.

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

**Interpretation:**
- Units: same as target variable
- "On average, our predictions are off by MAE units"
- Example: MAE = 10,000 on house prices means average error = $10,000

**When to use:**
- When you want a robust measure not dominated by outliers
- When all errors should contribute equally
- When the cost of error is proportional (not quadratic) to the error size

**Advantages:**
- ✅ Same units as target → interpretable
- ✅ Robust to outliers (linear penalty, not quadratic)
- ✅ Easy to understand: "average absolute error"

**Disadvantages:**
- ❌ Not differentiable at zero (harder to optimize directly)
- ❌ Doesn't penalize large errors as much (may not suit safety-critical applications)
- ❌ Scale-dependent

**MSE vs MAE Example:**
```
Predictions errors: [1, 1, 1, 10]

MAE  = (1 + 1 + 1 + 10) / 4 = 3.25
RMSE = √((1 + 1 + 1 + 100) / 4) = √25.75 ≈ 5.07

→ RMSE is much larger because the outlier (10) gets squared to 100.
  Choose MAE if that outlier shouldn't dominate the evaluation.
  Choose RMSE if that large error is genuinely important to penalize.
```

---

### 4. R² Score (Coefficient of Determination)

**Definition:** Measures the proportion of variance in the target variable that is explained by the model. It compares the model's predictions against the simplest baseline: always predicting the mean.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Where:

$$SS_{res} = \sum_{i=1}^{m}(y_i - \hat{y}_i)^2 \quad \text{(residual sum of squares — model error)}$$

$$SS_{tot} = \sum_{i=1}^{m}(y_i - \bar{y})^2 \quad \text{(total sum of squares — data variance)}$$

**Interpretation:**
- R² = 1.0: Model explains 100% of the variance (perfect fit, likely overfitting)
- R² = 0.85: Model explains 85% of the variance (good)
- R² = 0.0: Model is no better than predicting the mean
- R² < 0: Model is worse than predicting the mean (very poor)

![R² Values Illustration](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/R2values.svg/500px-R2values.svg.png)

*R² shows the proportion of variance explained. Higher R² = data points cluster tighter around the regression line.*

![Coefficient of Determination](https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Coefficient_of_Determination.svg/400px-Coefficient_of_Determination.svg.png)

*The blue squares represent residuals with respect to the regression line (SSres). The red squares represent residuals with respect to the mean (SStot). R² = 1 − (blue / red).*

**When to use:**
- Comparing models on the same dataset
- When you need a scale-independent measure of fit quality
- Quick summary of model explanatory power

**Advantages:**
- ✅ Scale-independent (0 to 1 range for reasonable models)
- ✅ Intuitive: "percentage of variance explained"
- ✅ Standard metric for regression comparison

**Disadvantages:**
- ❌ Always increases when adding more features (even irrelevant ones)
- ❌ Doesn't indicate if the model is biased
- ❌ Can be misleading with non-linear relationships
- ❌ Use **Adjusted R²** for multiple regression to account for number of features

**Adjusted R²:**

$$R^2_{adj} = 1 - \frac{(1 - R^2)(m - 1)}{m - n - 1}$$

Where $m$ = samples, $n$ = features. Penalizes adding irrelevant features.

---

### Evaluation Metrics Summary

| Metric | Units | Outlier Sensitive? | Interpretability | Best For |
|--------|-------|-------------------|------------------|----------|
| **MSE** | Squared | ❌ Very sensitive | Low (squared units) | Optimization / training |
| **RMSE** | Original | ❌ Sensitive | High | Reporting, comparing models |
| **MAE** | Original | ✅ Robust | High | Robust evaluation, outlier-heavy data |
| **R²** | Unitless | ❌ Sensitive | High | Comparing model explanatory power |

## Regularization Techniques

> **Regularization prevents overfitting** by adding a penalty term to the cost function that discourages large coefficients.
>
> Three main techniques: **Ridge (L2)**, **Lasso (L1)**, and **Elastic Net (L1+L2)**.
>
> See **[regularization_techniques.md](regularization_techniques.md)** for the full guide on each technique — including cost functions, how to apply them in Python, how to interpret results, how to choose hyperparameters, and advantages/disadvantages.

---

## Linear Regression Models: Estimation Methods and Cost Functions

Below is a summary of each model variant, its estimation method, and cost function.

### 1. Ordinary Least Squares (OLS) — Standard Linear Regression

**Definition:** OLS is the most basic form of linear regression. It finds the line (or hyperplane) that minimizes the sum of squared residuals between predicted and actual values. There is no penalty for coefficient size — the model is free to assign any weight to any feature.

**Estimation Method:** 
- **Normal Equation** (closed-form): $\theta = (X^T X)^{-1} X^T y$
- **Gradient Descent** (iterative): $\theta := \theta - \frac{\alpha}{m} X^T (X\theta - y)$

**Cost Function:**

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

This is pure MSE with no penalty term.

**Properties:**
- Unbiased estimator (gives the true parameters on average)
- Minimum variance among unbiased estimators (Gauss-Markov theorem)
- Prone to overfitting when many features or multicollinearity exists
- Baseline model — always start here

---

### 2. Ridge Regression (L2 Regularized)

**Definition:** OLS + L2 penalty. Shrinks coefficients to reduce model complexity and handle multicollinearity.

**Estimation Method:**
- **Normal Equation**: $\theta = (X^T X + \lambda I)^{-1} X^T y$
- **Gradient Descent**: $\theta := \theta - \alpha \left[\frac{1}{m} X^T(X\theta - y) + 2\lambda\theta\right]$

**Cost Function:**

$$J(\theta) = \frac{1}{2m} \sum (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

**Properties:**
- Biased estimator (trades small bias for large variance reduction)
- Always invertible ($\lambda I$ fixes singularity)
- Keeps all features

---

### 3. Lasso Regression (L1 Regularized)

**Definition:** OLS + L1 penalty. Shrinks coefficients and can eliminate features entirely.

**Estimation Method:**
- **No closed-form solution**
- **Coordinate Descent** (most common): Optimize one $\theta_j$ at a time, cycling through all features
- **Subgradient Descent**: $\theta := \theta - \alpha \left[\frac{1}{m} X^T(X\theta - y) + \lambda \cdot \text{sign}(\theta)\right]$

**Cost Function:**

$$J(\theta) = \frac{1}{2m} \sum (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

**Properties:**
- Biased estimator
- Performs feature selection (sparse solutions)
- Not differentiable at $\theta_j = 0$ (requires subgradient or coordinate descent)

---

### 4. Elastic Net (L1 + L2 Regularized)

**Definition:** OLS + combined L1 and L2 penalties. Balances feature selection with coefficient stability.

**Estimation Method:**
- **No closed-form solution**
- **Coordinate Descent** (most common)
- **Gradient Descent** with combined L1/L2 gradients

**Cost Function:**

$$J(\theta) = \frac{1}{2m} \sum (\hat{y}_i - y_i)^2 + r \cdot \lambda \sum |\theta_j| + \frac{(1-r)}{2} \cdot \lambda \sum \theta_j^2$$

**Properties:**
- Biased estimator
- Feature selection (from L1) + stability (from L2)
- Groups correlated features together

---

### Model Comparison Summary

| Model | Penalty | Estimation | Feature Selection | Bias | Variance |
|-------|---------|------------|-------------------|------|----------|
| **OLS** | None | Normal Eq. / GD | ❌ | Lowest | Highest |
| **Ridge** | $\lambda\sum\theta_j^2$ | Normal Eq. / GD | ❌ | Low | Lower |
| **Lasso** | $\lambda\sum\|\theta_j\|$ | Coordinate Descent | ✅ | Low | Lower |
| **Elastic Net** | $\lambda(r\sum\|\theta_j\| + (1\text{-}r)\sum\theta_j^2)$ | Coordinate Descent | ✅ | Low | Lower |

![Bias-Variance Tradeoff](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/500px-Bias_and_variance_contributing_to_total_error.svg.png)

*Total Error = Bias² + Variance + Irreducible Error. Regularization moves us from the overfitting zone (right) toward the sweet spot where total error is minimized. High $\lambda$ → underfitting (high bias), low $\lambda$ → overfitting (high variance).*

---

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
| $X^T X$ is singular | Gradient Descent or Regularization |

## Time Complexity

- **Normal Equation:** $O(n^3)$ due to matrix inversion
- **Gradient Descent:** $O(kmn)$ where $k$ = iterations, $m$ = samples, $n$ = features

---

**Remember:** Linear regression is simple yet powerful. Understanding both approaches helps you choose the right tool for the problem at hand!
