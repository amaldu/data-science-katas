# Logistic Regression Cheatsheet

## What is Logistic Regression?

Logistic Regression is a supervised learning algorithm used for **binary classification**. Despite its name, it is a **classification** algorithm, not a regression one. It predicts the probability that an input belongs to a certain class using the **sigmoid function**.

### Core Idea
Apply a linear function, then pass it through the sigmoid to get a probability:
```
z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ    (linear part)
ŷ = σ(z) = 1 / (1 + e⁻ᶻ)               (sigmoid squashes to [0, 1])
```

### Decision Rule
```
If ŷ >= 0.5 → Class 1
If ŷ <  0.5 → Class 0
```
The threshold 0.5 can be adjusted depending on the problem.

## Sigmoid Function

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Properties:**
- Output range: (0, 1)
- σ(0) = 0.5
- As z → +∞, σ(z) → 1
- As z → −∞, σ(z) → 0
- Derivative: σ'(z) = σ(z) · (1 − σ(z))
- S-shaped curve (monotonically increasing)

## Hypothesis Function

**In matrix form:**
```
z = Xθ
ŷ = σ(Xθ) = 1 / (1 + e^(−Xθ))
```

Where:
- `X` = feature matrix (m × n), with bias column of ones
- `θ` = parameter vector [θ₀, θ₁, ..., θₙ]ᵀ
- `ŷ` = predicted probabilities

## Cost Function (Binary Cross-Entropy / Log Loss)

**Why not use MSE?**  
MSE with sigmoid creates a **non-convex** cost function with many local minima. Log loss is convex and works with gradient descent.

**Log Loss formula:**
```
J(θ) = −(1/m) Σᵢ₌₁ᵐ [yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ)]
```

**Intuition:**
- When y = 1: cost = −log(ŷ) → penalizes low predictions
- When y = 0: cost = −log(1 − ŷ) → penalizes high predictions
- Perfect prediction → cost = 0
- Wrong prediction → cost → ∞

## Gradient Descent for Logistic Regression

**Gradient (partial derivative of J with respect to θⱼ):**
```
∂J/∂θⱼ = (1/m) Σᵢ₌₁ᵐ (ŷᵢ − yᵢ) xᵢⱼ
```

**Vectorized gradient:**
```
∇J(θ) = (1/m) Xᵀ(σ(Xθ) − y)
```

**Update rule:**
```
θ := θ − α · ∇J(θ)
```

> Note: The gradient formula looks identical to linear regression, but ŷ = σ(Xθ) instead of ŷ = Xθ.

## Key Differences: Linear vs Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| **Task** | Regression (continuous) | Classification (discrete) |
| **Output** | Any real number | Probability [0, 1] |
| **Hypothesis** | ŷ = Xθ | ŷ = σ(Xθ) |
| **Cost function** | MSE | Binary Cross-Entropy |
| **Normal equation** | Yes | No (no closed-form) |
| **Optimization** | GD or Normal Eq. | Gradient Descent only |
| **Decision boundary** | N/A | Linear hyperplane |

## Decision Boundary

The decision boundary is the surface where σ(Xθ) = 0.5, which means Xθ = 0.

```
θ₀ + θ₁x₁ + θ₂x₂ = 0
```

- Linear decision boundary for logistic regression
- Can create non-linear boundaries by adding polynomial features

## Multiclass Classification

Logistic regression is inherently binary, but can be extended:

### One-vs-Rest (OvR / OvA)
- Train K separate binary classifiers (one per class)
- Each classifier: "class k" vs "everything else"
- Predict class with highest probability

### One-vs-One (OvO)
- Train K(K−1)/2 classifiers (one per class pair)
- Each classifier distinguishes between two classes
- Predict by majority vote

### Softmax Regression (Multinomial)
- Generalizes logistic regression to K classes directly
- Uses softmax function instead of sigmoid:
```
P(y = k) = e^(θₖᵀx) / Σⱼ₌₁ᴷ e^(θⱼᵀx)
```
- Cost function: Categorical Cross-Entropy

## Regularization

### L2 Regularization (Ridge)
```
J(θ) = −(1/m) Σ[y log(ŷ) + (1−y) log(1−ŷ)] + (λ/2m) Σⱼ₌₁ⁿ θⱼ²
```
- Prevents overfitting
- Shrinks coefficients toward zero

### L1 Regularization (Lasso)
```
J(θ) = −(1/m) Σ[y log(ŷ) + (1−y) log(1−ŷ)] + (λ/m) Σⱼ₌₁ⁿ |θⱼ|
```
- Feature selection (drives some weights to exactly zero)
- Creates sparse models

## Evaluation Metrics for Classification

### Confusion Matrix
```
                  Predicted
                  Pos    Neg
Actual  Pos  |   TP   |  FN  |
        Neg  |   FP   |  TN  |
```

### Key Metrics
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)           — "Of predicted positives, how many are correct?"
Recall    = TP / (TP + FN)           — "Of actual positives, how many did we find?"
F1 Score  = 2 · (Precision · Recall) / (Precision + Recall)
```

### When to Use Each
- **Accuracy**: Balanced classes
- **Precision**: When false positives are costly (spam detection)
- **Recall**: When false negatives are costly (disease detection)
- **F1 Score**: Imbalanced classes, need balance of precision and recall

### ROC-AUC
- **ROC Curve**: True Positive Rate vs False Positive Rate at various thresholds
- **AUC**: Area Under the ROC Curve (0.5 = random, 1.0 = perfect)
- Threshold-independent evaluation metric

## Assumptions of Logistic Regression

1. **Binary or ordinal outcome** (for standard logistic regression)
2. **Independence** of observations
3. **Little or no multicollinearity** among features
4. **Linear relationship** between features and log-odds
5. **Large sample size** — needs sufficient data per class

## Python Implementation Summary

**From scratch:**
```python
# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost
def cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1/m) * (y @ np.log(h) + (1-y) @ np.log(1-h))

# Gradient
def gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return (1/m) * X.T @ (h - y)

# Update
theta = theta - learning_rate * gradient(X, y, theta)
```

**Using Scikit-learn:**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Common Pitfalls

1. **Using MSE** as cost function → non-convex, won't converge properly
2. **Not scaling features** → slow/poor convergence with gradient descent
3. **Ignoring class imbalance** → model predicts majority class
4. **Ignoring multicollinearity** → unstable coefficients
5. **Choosing wrong threshold** → always evaluate multiple thresholds

## Quick Reference

| Component | Formula |
|-----------|---------|
| Sigmoid | σ(z) = 1 / (1 + e⁻ᶻ) |
| Hypothesis | ŷ = σ(Xθ) |
| Cost (Log Loss) | J = −(1/m) Σ[y log(ŷ) + (1−y) log(1−ŷ)] |
| Gradient | ∇J = (1/m) Xᵀ(ŷ − y) |
| Update | θ := θ − α · ∇J |
| Decision | class 1 if σ(Xθ) ≥ 0.5 |

---

**Remember:** Logistic Regression is simple, interpretable, and surprisingly effective. It's often the first baseline for any classification problem!
