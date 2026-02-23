# Logistic Regression Cheatsheet

## What is Logistic Regression?

Logistic Regression is a supervised learning algorithm used for **binary classification**. It predicts the probability that an input belongs to a certain class using the **sigmoid function**.



The core Idea is to apply a linear function, then pass it through the sigmoid to get a probability:
```
z = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ    (linear part)
ŷ = σ(z) = 1 / (1 + e⁻ᶻ)               (sigmoid squashes to [0, 1])
```

### Decision Rule
```
If ŷ >= 0.5 → Class 1
If ŷ <  0.5 → Class 0
```

The default threshold is 0.5, but it **should be adjusted** depending on the problem. The threshold directly controls the trade-off between precision and recall.

**How adjusting the threshold works:**

| Threshold | Effect | Precision | Recall | Use when... |
|:---:|:---|:---:|:---:|:---|
| **Low** (e.g., 0.2) | Model predicts positive more often | Lower | Higher | Missing a positive is very costly (disease, fraud) |
| **Default** (0.5) | Balanced | Balanced | Balanced | No strong preference between error types |
| **High** (e.g., 0.8) | Model predicts positive only when very confident | Higher | Lower | False alarms are very costly (spam, criminal conviction) |

**Criteria for choosing the threshold:**

**1. Cost of errors — the most important factor.**
Ask: "Which mistake is worse — a false positive or a false negative?"

- **False negatives are worse** (missing cancer, missing fraud) → **lower the threshold** (e.g., 0.2–0.4). You'd rather flag more cases and investigate, even if some turn out to be false alarms.
- **False positives are worse** (blocking a legitimate email, unnecessary surgery) → **raise the threshold** (e.g., 0.6–0.8). You only want to act when the model is highly confident.
- **Both are equally bad** → keep the default 0.5 or optimize for F1 score.

**2. Class imbalance.**
When the positive class is rare (e.g., 1% fraud), the default 0.5 threshold often predicts almost everything as negative. Lowering the threshold helps the model detect more of the rare positive class.

**3. Use the Precision-Recall curve.**
Plot precision and recall at every threshold and pick the point that best fits your business requirement:

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

# Find threshold that gives at least 90% recall
for p, r, t in zip(precisions, recalls, thresholds):
    if r >= 0.90:
        print(f"Threshold: {t:.3f}, Precision: {p:.3f}, Recall: {r:.3f}")
        break
```

**4. Optimize for a specific metric.**
Choose the threshold that maximizes the metric you care about:

```python
from sklearn.metrics import f1_score
import numpy as np

# Find threshold that maximizes F1 score
best_f1, best_threshold = 0, 0.5
for t in np.arange(0.1, 0.9, 0.01):
    y_pred = (y_scores >= t).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.3f}")
```

**5. Use the ROC curve.**
The Youden's J statistic picks the threshold that maximizes $\text{TPR} - \text{FPR}$ (the point on the ROC curve farthest from the diagonal):

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"Optimal threshold (Youden's J): {best_threshold:.3f}")
```

**Key takeaway:** The threshold is a **business decision**, not a purely technical one. Always involve domain experts when choosing it, and never assume 0.5 is optimal.

## Sigmoid Function

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

![Sigmoid Function](../images/15_sigmoid_function.svg)

*The sigmoid (logistic) function maps any real-valued input to the range (0, 1). At z = 0 it outputs 0.5; large positive z approaches 1; large negative z approaches 0.*

**Properties:**
- Output range: (0, 1)
- σ(0) = 0.5
- As z → +∞, σ(z) → 1
- As z → −∞, σ(z) → 0
- Derivative: σ'(z) = σ(z) · (1 − σ(z))
- S-shaped curve (monotonically increasing)

## Estimation Method

Logistic Regression uses **Maximum Likelihood Estimation (MLE)** to find the optimal parameters $\theta$.

**What is MLE?** Logistic Regression finds the parameters that **maximize the probability of observing the actual data**. In other words, MLE asks: "What values of $\theta$ make the observed outcomes (0s and 1s) most likely?"

**How it works:**
1. For each training sample, the model predicts a probability $\hat{p}_i = \sigma(X_i \theta)$.
2. If the true label is $y_i = 1$, we want $\hat{p}_i$ to be as close to 1 as possible.
3. If the true label is $y_i = 0$, we want $\hat{p}_i$ to be as close to 0 as possible.
4. The **likelihood** is the product of all these individual probabilities across the dataset.
5. MLE finds $\theta$ that maximizes this likelihood.

**The Likelihood Function:**

$$L(\theta) = \prod_{i=1}^{m} \hat{p}_i^{y_i} \cdot (1 - \hat{p}_i)^{1-y_i}$$

Since products are numerically unstable and hard to differentiate, we take the **log** (converting products to sums):

$$\log L(\theta) = \sum_{i=1}^{m} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**Maximizing** the log-likelihood is equivalent to **minimizing** the negative log-likelihood — which is exactly the **Binary Cross-Entropy** cost function:

$$J(\theta) = -\frac{1}{m} \log L(\theta)$$

This is why the cost function for logistic regression has its specific form — it comes directly from the MLE principle.

Logistic Regression must be solved **iteratively** using optimization algorithms (Gradient Descent, Newton's Method, L-BFGS).

## Hypothesis Function

The hypothesis function is the **complete prediction model** — it takes input features and produces a probability that the sample belongs to class 1. It has two stages: a linear combination followed by the sigmoid transformation.

**Stage 1 — Linear combination (same as Linear Regression):**

$$z = X\theta = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

This computes a raw score $z$ (called the **logit** or **log-odds**) that can range from $-\infty$ to $+\infty$. A positive $z$ means the model leans toward class 1; a negative $z$ leans toward class 0.

**Stage 2 — Sigmoid transformation (converts score to probability):**

$$\hat{y} = \sigma(z) = \sigma(X\theta) = \frac{1}{1 + e^{-X\theta}}$$

The sigmoid squashes the unbounded score $z$ into the range $(0, 1)$, giving it a valid probabilistic interpretation: $\hat{y} = P(y = 1 \mid X, \theta)$.

**In matrix form:**

$$z = X\theta \qquad \hat{y} = \sigma(X\theta)$$

**Where each component means:**
- $X$ = feature matrix $(m \times n)$, with a bias column of ones prepended. Each row is one sample, each column is one feature.
- $\theta$ = parameter vector $[\theta_0, \theta_1, \ldots, \theta_n]^T$. These are the weights the model learns during training. $\theta_0$ is the intercept (bias).
- $z = X\theta$ = the **logit** — a linear score for each sample. Positive = more likely class 1, negative = more likely class 0.
- $\sigma(\cdot)$ = the sigmoid function, applied element-wise. Converts each logit into a probability.
- $\hat{y}$ = predicted probabilities, one per sample. $\hat{y}_i \in (0, 1)$.

**Example:** For a model with 2 features and learned parameters $\theta = [−1.5, \; 0.8, \; 1.2]$:

$$z = -1.5 + 0.8 \cdot x_1 + 1.2 \cdot x_2$$

- If $x_1 = 3, x_2 = 2$: $z = -1.5 + 2.4 + 2.4 = 3.3$ → $\hat{y} = \sigma(3.3) = 0.964$ → predict class 1
- If $x_1 = 0, x_2 = 0$: $z = -1.5$ → $\hat{y} = \sigma(-1.5) = 0.182$ → predict class 0

## Cost Function (Binary Cross-Entropy / Log Loss)

## Key Assumptions

Logistic Regression makes several assumptions. Violating them can lead to unreliable coefficient estimates, poor predictions, or misleading significance tests.

---

### 1. Binary or Ordinal Outcome

**What it means:** The dependent variable (target) must be binary (0/1) for standard logistic regression, or ordinal for ordinal logistic regression.

**Why it matters:** The sigmoid function maps to the range $(0, 1)$, which represents the probability of belonging to one of two classes. If the target has more than two unordered categories, use **Softmax Regression** (multinomial) or **One-vs-Rest** instead.

---

### 2. Independence of Observations

**What it means:** Each data point must be independent of the others — the outcome of one observation should not influence or predict the outcome of another.

**Why it matters:** Correlated observations (e.g., repeated measurements on the same patient, time series data) violate this assumption. The model underestimates the standard errors of coefficients, making significance tests unreliable (p-values too small, confidence intervals too narrow).

**What to do if violated:** Use generalized estimating equations (GEE), mixed-effects logistic regression, or time-series specific models.

---

### 3. Little or No Multicollinearity

**What it means:** The independent features should not be highly correlated with each other. Moderate correlation is acceptable, but very high correlation (e.g., $|r| > 0.9$) causes problems.

**Why it matters:** Multicollinearity inflates the variance of coefficient estimates, making them unstable and hard to interpret. The model still predicts well overall, but individual coefficients become unreliable — small changes in data can flip their signs.

**How to detect:** Compute the Variance Inflation Factor (VIF) for each feature. VIF $> 5-10$ indicates problematic multicollinearity.

**What to do:** Remove one of the correlated features, apply PCA to reduce dimensionality, or use regularization (L1/L2) which stabilizes the estimates.

---

### 4. Linear Relationship Between Features and Log-Odds

**What it means:** Logistic Regression assumes that the **log-odds** (logit) of the outcome is a linear function of the features:

$$\log\left(\frac{p}{1-p}\right) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

This does NOT mean the relationship between features and probability is linear — the sigmoid makes it non-linear. But the relationship between features and **log-odds** must be linear.

**Why it matters:** If the true relationship between a feature and the log-odds is curved (e.g., U-shaped), the model will misrepresent it, leading to biased coefficients and poor predictions.

**How to detect:** Plot each continuous feature against the log-odds of the outcome. If the relationship is not approximately linear, the assumption is violated.

**What to do:** Apply transformations to the feature (e.g., $\log(x)$, $x^2$), add polynomial terms, or use a more flexible model (tree-based, neural network).

---

### 5. Large Sample Size

**What it means:** Logistic Regression requires a sufficiently large number of samples, especially relative to the number of features and the rarity of the minority class.

**Why it matters:** With small samples, MLE can produce unreliable estimates — coefficients may be extreme, standard errors inflated, and the model may not converge. A common rule of thumb is at least **10–20 events per predictor variable** (EPV), where "events" refers to the count of the minority class.

**What to do if sample is small:** Reduce the number of features, use regularization (L1/L2), apply penalized MLE (Firth's method), or collect more data.

---

### Assumptions Summary

| Assumption | What Breaks If Violated | Severity |
|:---|:---|:---|
| **Binary outcome** | Model is fundamentally wrong | High |
| **Independence** | Standard errors, p-values unreliable | High |
| **No multicollinearity** | Unstable coefficients, hard to interpret | Medium |
| **Linear log-odds** | Biased coefficients, poor predictions | Medium-High |
| **Large sample size** | Extreme estimates, non-convergence | Medium |

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


## Multiclass Classification

Logistic regression is inherently binary, but can be extended:

### One-vs-Rest (OvR / OvA)
- Train K separate binary classifiers (one per class)
- Each classifier: "class k" vs "everything else"
- Predict class with highest probability
- **Time Complexity:** Training: $O(K \cdot kmn)$ — train $K$ separate models. Prediction: $O(Kn)$ — evaluate all $K$ classifiers.

### One-vs-One (OvO)
- Train K(K−1)/2 classifiers (one per class pair)
- Each classifier distinguishes between two classes
- Predict by majority vote
- **Time Complexity:** Training: $O(\frac{K(K-1)}{2} \cdot km'n)$ — train $K(K-1)/2$ models, each on a subset of $m' < m$ samples. Prediction: $O(\frac{K(K-1)}{2} \cdot n)$.

### Softmax Regression (Multinomial)
- Generalizes logistic regression to K classes directly
- Uses softmax function instead of sigmoid:
```
P(y = k) = e^(θₖᵀx) / Σⱼ₌₁ᴷ e^(θⱼᵀx)
```
- Cost function: Categorical Cross-Entropy
- **Time Complexity:** Training: $O(kmKn)$ — one gradient step is $O(mKn)$ since $\Theta$ is an $n \times K$ matrix. Prediction: $O(Kn)$ — compute $K$ scores and softmax.

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

## Evaluation Metrics

> For a full guide on classification metrics, see **[classification_metrics_cheatsheet.md](classification_metrics_cheatsheet.md)** — covering Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log Loss, Cohen's Kappa, and multiclass extensions.

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


1. **Not scaling features** → slow/poor convergence with gradient descent
2. **Ignoring class imbalance** → model predicts majority class
3. **Ignoring multicollinearity** → unstable coefficients
4. **Choosing wrong threshold** → always evaluate multiple thresholds

