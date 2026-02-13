# Support Vector Machines (SVMs)

Support Vector Machines are **supervised learning models** that find the optimal hyperplane separating classes by maximizing the margin between the closest data points of each class. They are among the most robust and theoretically grounded classifiers in machine learning.

**Core idea:** Instead of just finding *any* boundary that separates the classes, SVMs find the boundary that is **as far as possible** from the nearest training points of each class. This maximum-margin property gives SVMs strong generalization.

---

## Linear SVM Classification

A Linear SVM finds a hyperplane $\mathbf{w} \cdot \mathbf{x} + b = 0$ that separates two classes with the **largest possible margin**.

**Decision function:**

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

- If $f(\mathbf{x}) \geq 0$ → class 1
- If $f(\mathbf{x}) < 0$ → class 0 (or class -1)

**The margin** is the distance between the two parallel hyperplanes $\mathbf{w} \cdot \mathbf{x} + b = +1$ and $\mathbf{w} \cdot \mathbf{x} + b = -1$:

$$\text{margin} = \frac{2}{\|\mathbf{w}\|}$$

Maximizing the margin is equivalent to **minimizing** $\|\mathbf{w}\|$ (or $\frac{1}{2}\|\mathbf{w}\|^2$ for mathematical convenience).

**Hard Margin Classification** requires that all instances are correctly classified and lie outside the margin. This only works if the data is **perfectly linearly separable** and is very sensitive to outliers.

```python
from sklearn.svm import SVC

# Hard margin: very large C means almost no tolerance for misclassification
model = SVC(kernel='linear', C=1e6)
model.fit(X_train, y_train)
```

---

## Soft Margin Classification

In the real world, data is rarely perfectly separable. **Soft margin** SVMs allow some misclassifications by introducing **slack variables** $\xi_i \geq 0$ that measure how much each instance violates the margin.

**Optimization objective:**

$$\min_{\mathbf{w}, b, \xi} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \xi_i$$

Subject to:

$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i \quad \text{and} \quad \xi_i \geq 0$$

**The hyperparameter $C$ controls the tradeoff:**

| $C$ Value | Behavior | Risk |
|-----------|----------|------|
| **Large $C$** | Few margin violations allowed — narrow margin, fits training data closely | Overfitting |
| **Small $C$** | Many margin violations allowed — wide margin, more regularization | Underfitting |

**Think of $C$ as the inverse of regularization strength:** large $C$ = low regularization, small $C$ = high regularization.

```python
from sklearn.svm import LinearSVC

# Soft margin with moderate regularization
model = LinearSVC(C=1.0, loss='hinge', max_iter=5000)
model.fit(X_train, y_train)
```

**When to use:**
- Default approach for linear SVM classification
- Always prefer soft margin over hard margin in practice (robust to outliers and noise)

---

## Nonlinear SVM Classification

When data is not linearly separable, you need to map it into a higher-dimensional space where a linear separator can be found.

### Polynomial Features (Explicit Mapping)

One approach is to explicitly add polynomial features, then apply a linear SVM:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

poly_svm = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=10, loss="hinge", max_iter=5000))
])
poly_svm.fit(X_train, y_train)
```

**Problem:** For high-degree polynomials or many features, the number of generated features **explodes combinatorially**, making this approach slow and memory-intensive.

**Solution:** The **kernel trick**.

---

### Polynomial Kernel

Instead of explicitly computing all polynomial feature combinations, the **polynomial kernel** computes the dot product in the higher-dimensional space *implicitly*:

$$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \, \mathbf{x}_i \cdot \mathbf{x}_j + r)^d$$

Where:
- $d$ = polynomial degree
- $\gamma$ = controls the influence of each training instance
- $r$ = coef0, shifts the kernel (independent term)

```python
from sklearn.svm import SVC

# Polynomial kernel of degree 3
model = SVC(kernel='poly', degree=3, coef0=1, C=5)
model.fit(X_train, y_train)
```

**Hyperparameters to tune:**

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `degree` | More complex decision boundary, risk of overfitting |
| `coef0` | Higher-degree terms get more influence |
| `C` | Less regularization, tighter fit to training data |

**When to use:**
- Data has polynomial relationships
- You know the approximate degree of the relationship
- Moderate-sized datasets (kernel SVMs scale poorly with $m$)

---

### Adding Similarity Features

Another approach to nonlinearity is computing **similarity features** using a similarity function (e.g., Gaussian RBF) with respect to **landmark** points.

**Idea:** For each instance $\mathbf{x}$, create a new feature for each landmark $\ell$:

$$\phi(\mathbf{x}, \ell) = \exp\left(-\gamma \|\mathbf{x} - \ell\|^2\right)$$

This transforms the feature space so that instances close to a landmark have a feature value near 1, and instances far from it have a value near 0.

**How to choose landmarks?** The simplest approach is to create a landmark at every training instance — which is exactly what the RBF kernel does implicitly.

---

### Gaussian RBF Kernel

The **Radial Basis Function (RBF)** kernel is the most popular kernel for nonlinear SVMs:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

- $\gamma$ large → each instance has a narrow influence (high variance, complex boundary)
- $\gamma$ small → each instance has a wide influence (high bias, smoother boundary)

**$\gamma$ and $C$ interact together:**

| $\gamma$ | $C$ | Result |
|-----------|-----|--------|
| High | High | Overfitting — complex boundary, few violations allowed |
| High | Low | Moderate — complex boundary but more violations allowed |
| Low | High | Moderate — smooth boundary but few violations allowed |
| Low | Low | Underfitting — smooth boundary, many violations allowed |

```python
from sklearn.svm import SVC

# RBF kernel (default)
model = SVC(kernel='rbf', gamma=0.1, C=1.0)
model.fit(X_train, y_train)
```

**When to use:**
- Default choice for nonlinear problems
- When you don't know the structure of the nonlinearity
- Works well on moderate-sized datasets (up to tens of thousands of instances)

**Rule of thumb:** Start with RBF kernel. Only switch to polynomial if you have domain knowledge suggesting polynomial relationships, or to linear if the dataset is very large or high-dimensional.

---

## Computational Complexity

| Algorithm | Training Complexity | Prediction Complexity | Best For |
|-----------|--------------------|-----------------------|----------|
| `LinearSVC` | $O(m \times n)$ | $O(n)$ | Large datasets, many features |
| `SVC` (linear kernel) | $O(m^2 \times n)$ to $O(m^3 \times n)$ | $O(n_{sv} \times n)$ | Small to medium datasets |
| `SVC` (RBF kernel) | $O(m^2 \times n)$ to $O(m^3 \times n)$ | $O(n_{sv} \times n)$ | Non-linear, medium datasets |
| `SGDClassifier` (hinge) | $O(m \times n)$ | $O(n)$ | Very large datasets, online learning |

Where $m$ = samples, $n$ = features, $n_{sv}$ = number of support vectors.

**Key takeaway:** Kernel SVMs (`SVC`) scale **poorly** with the number of training instances due to $O(m^2)$ to $O(m^3)$ training time. For datasets larger than ~50,000 instances, prefer `LinearSVC` or `SGDClassifier`.

```python
# For large datasets — use LinearSVC (much faster)
from sklearn.svm import LinearSVC
model = LinearSVC(C=1.0, loss='hinge', max_iter=10000)

# For very large datasets — use SGDClassifier with hinge loss (online SVM)
from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='hinge', alpha=1/(m*C), max_iter=1000)

# For small/medium datasets with nonlinearity — use SVC with kernel
from sklearn.svm import SVC
model = SVC(kernel='rbf', gamma='scale', C=1.0)
```

---

## SVM Regression

SVMs can also be used for regression (**Support Vector Regression — SVR**). Instead of maximizing the margin between classes, SVR tries to fit as many instances as possible **within a margin** of width $\varepsilon$ around the prediction, while keeping the model as flat as possible.

**Key difference from classification:**
- **Classification:** maximize the margin, keep instances *outside* the margin
- **Regression:** keep instances *inside* the margin (within $\varepsilon$ of the prediction)

**Optimization objective:**

$$\min_{\mathbf{w}, b, \xi, \xi^*} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{m}(\xi_i + \xi_i^*)$$

Subject to:

$$y_i - (\mathbf{w} \cdot \mathbf{x}_i + b) \leq \varepsilon + \xi_i$$
$$(\mathbf{w} \cdot \mathbf{x}_i + b) - y_i \leq \varepsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

**Hyperparameters:**

| Parameter | Effect |
|-----------|--------|
| $\varepsilon$ | Width of the "tube" — larger $\varepsilon$ means more tolerance, fewer support vectors, smoother model |
| $C$ | Regularization — large $C$ penalizes points outside the tube more, tighter fit |

```python
from sklearn.svm import SVR, LinearSVR

# Linear SVR (fast, for large datasets)
model = LinearSVR(epsilon=1.5, C=1.0, max_iter=10000)
model.fit(X_train, y_train)

# Nonlinear SVR with RBF kernel (for complex relationships)
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_train, y_train)
```

**When to use SVR:**
- When you need robustness to outliers (the $\varepsilon$-insensitive loss ignores small errors)
- When the dataset is small to medium-sized
- When you need nonlinear regression with kernel flexibility

---

## Under the Hood

### Decision Function and Predictions

For a trained SVM, the decision function for a new instance $\mathbf{x}$ is:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^{m} \alpha_i \, y_i \, (\mathbf{x}_i \cdot \mathbf{x}) + b$$

Where:
- $\alpha_i$ are the **Lagrange multipliers** (dual coefficients) — only nonzero for **support vectors**
- $y_i \in \{-1, +1\}$ are the class labels
- $\mathbf{x}_i$ are the training instances

**Classification rule:** $\hat{y} = \text{sign}(f(\mathbf{x}))$

The key insight: the decision function depends **only on the support vectors** (instances with $\alpha_i > 0$), not on all training data. This makes predictions efficient even if the training set was large.

---

### Training Objective

The SVM training objective in **primal form** is:

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \max\left(0, 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i + b)\right)$$

This combines:
1. **Margin maximization** — the $\frac{1}{2}\|\mathbf{w}\|^2$ term (smaller $\|\mathbf{w}\|$ = wider margin)
2. **Hinge loss** — the $\max(0, 1 - y_i f(\mathbf{x}_i))$ term penalizes violations

The hinge loss is zero when an instance is correctly classified and on the correct side of the margin. It increases linearly as the instance moves to the wrong side.

---

### Quadratic Programming

The SVM optimization problem is a **Quadratic Programming (QP)** problem — a convex optimization problem with a quadratic objective function and linear constraints.

**General QP form:**

$$\min_{\mathbf{p}} \quad \frac{1}{2}\mathbf{p}^T \mathbf{H} \mathbf{p} + \mathbf{f}^T \mathbf{p}$$

Subject to: $\mathbf{A} \mathbf{p} \leq \mathbf{b}$

For the **soft margin linear SVM** (hard margin when $C \to \infty$):

| QP Parameter | SVM Meaning | Dimensions |
|--------------|-------------|------------|
| $\mathbf{p}$ | Contains $\mathbf{w}$, $b$, and slack variables $\xi_i$ | $(n + 1 + m) \times 1$ |
| $\mathbf{H}$ | Identity matrix for $\mathbf{w}$ part, zeros elsewhere | $(n+1+m) \times (n+1+m)$ |
| $\mathbf{f}$ | $C$ for the slack variables, 0 for $\mathbf{w}$ and $b$ | $(n+1+m) \times 1$ |
| $\mathbf{A}$ | Encodes the classification constraints | $2m \times (n+1+m)$ |
| $\mathbf{b}$ | $-1$ for classification constraints, $0$ for slack constraints | $2m \times 1$ |

**Why QP matters:** Because the SVM problem is convex QP, it is guaranteed to have a **unique global optimum** — no local minima. Any QP solver will find the optimal solution.

---

### The Dual Problem

Instead of solving the primal problem directly, we can solve its **dual form** using Lagrange multipliers:

$$\max_{\alpha} \quad \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m}\sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$

Subject to: $0 \leq \alpha_i \leq C$ and $\sum_{i=1}^{m} \alpha_i y_i = 0$

**Primal vs. Dual — when to use which:**

| Aspect | Primal | Dual |
|--------|--------|------|
| **Variables** | $n + 1$ (features + bias) | $m$ (one per training instance) |
| **Better when** | $m \gg n$ (many instances, few features) | $n \gg m$ (many features, few instances) |
| **Kernel trick** | Not applicable | Enables the kernel trick |
| **Solver** | `LinearSVC` (uses liblinear, primal) | `SVC` (uses libsvm, dual) |

**Rule of thumb:**
- **Millions of instances, hundreds of features** → solve the **primal** (use `LinearSVC`)
- **Thousands of instances, need kernels** → solve the **dual** (use `SVC`)

---

### Kernelized SVMs

The kernel trick is what makes the dual form powerful. Notice that the dual problem only involves **dot products** $\mathbf{x}_i \cdot \mathbf{x}_j$. We can replace every dot product with a **kernel function**:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$$

Where $\phi$ is a (potentially infinite-dimensional) feature mapping. The kernel computes the dot product in the transformed space **without ever computing** $\phi(\mathbf{x})$ explicitly.

**Common kernels:**

| Kernel | Formula | When to Use |
|--------|---------|-------------|
| **Linear** | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j$ | Linearly separable data, large datasets |
| **Polynomial** | $(\gamma \, \mathbf{x}_i \cdot \mathbf{x}_j + r)^d$ | Polynomial relationships, known degree |
| **RBF (Gaussian)** | $\exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | Default for nonlinear, unknown structure |
| **Sigmoid** | $\tanh(\gamma \, \mathbf{x}_i \cdot \mathbf{x}_j + r)$ | Rarely used (not always a valid kernel) |

**Mercer's Theorem:** A function $K$ is a valid kernel if and only if it satisfies **Mercer's condition** — the kernel matrix must be positive semi-definite for any set of inputs. All kernels above (except sigmoid in some parameter ranges) satisfy this.

---

### Online SVMs

Standard SVM solvers (like `libsvm`) are **batch** algorithms — they require the entire dataset in memory. For very large datasets or streaming data, **online SVMs** are needed.

**SGDClassifier with hinge loss** implements an online, approximate SVM using Stochastic Gradient Descent:

$$\text{SGD objective:} \quad \frac{\alpha}{2}\|\mathbf{w}\|^2 + \frac{1}{m}\sum_{i=1}^{m} \max(0, 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i + b))$$

Where $\alpha$ is the regularization parameter (equivalent to $\frac{1}{mC}$ in the standard SVM formulation).

```python
from sklearn.linear_model import SGDClassifier

# Online SVM using SGD
model = SGDClassifier(loss='hinge', alpha=0.001, max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# For streaming data — partial_fit for online learning
model = SGDClassifier(loss='hinge', alpha=0.001)
for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
```

**Key properties of online SVMs:**

| Property | Batch SVM (`SVC`) | Online SVM (`SGDClassifier`) |
|----------|-------------------|------------------------------|
| **Training** | Full dataset at once | One sample/batch at a time |
| **Convergence** | Exact (QP solver) | Approximate (SGD) |
| **Scaling** | $O(m^2)$ to $O(m^3)$ | $O(m \times n)$ per epoch |
| **Kernel support** | Yes (all kernels) | No (linear only) |
| **Streaming data** | No | Yes (`partial_fit`) |
| **Memory** | $O(m^2)$ for kernel matrix | $O(n)$ for weight vector |

---

## Pros, Cons, and Usage

### Advantages

- **Strong generalization** — maximum margin principle provides theoretical guarantees against overfitting
- **Effective in high-dimensional spaces** — works well even when $n > m$ (more features than samples)
- **Kernel flexibility** — can model complex nonlinear decision boundaries without explicit feature mapping
- **Robust to overfitting** (with proper $C$) — especially in high-dimensional spaces with clear margin
- **Global optimum guaranteed** — convex optimization means no local minima
- **Memory efficient** — only support vectors are stored for predictions
- **Versatile** — works for classification, regression, and outlier detection

### Disadvantages

- **Slow on large datasets** — kernel SVMs scale $O(m^2)$ to $O(m^3)$; impractical for $m > 50\text{k}$ with kernels
- **Sensitive to feature scaling** — SVMs compute distances, so features on different scales distort the margin. **Always scale your features.**
- **No native probability estimates** — `SVC` can output probabilities via Platt scaling (`probability=True`), but this is slow and sometimes poorly calibrated
- **Difficult to interpret** — unlike linear models or decision trees, the learned boundary is not easily explainable (especially with kernels)
- **Hyperparameter tuning required** — $C$, $\gamma$, kernel choice all significantly affect performance
- **Not great for noisy data with overlapping classes** — soft margin helps, but SVMs prefer clean margins
- **Binary by nature** — multi-class requires strategies like One-vs-One (OvO) or One-vs-All (OvR)

### When to Use SVMs

| Scenario | Recommendation |
|----------|---------------|
| Small to medium dataset (< 50k samples) with nonlinear boundaries | **SVC with RBF kernel** |
| Large dataset (> 50k samples), linear problem | **LinearSVC or SGDClassifier** |
| High-dimensional sparse data (e.g., text classification) | **LinearSVC** (very effective) |
| Need nonlinear regression on small datasets | **SVR with RBF kernel** |
| Online / streaming data | **SGDClassifier with hinge loss** |
| Need interpretable model | **Consider other models** (logistic regression, decision trees) |
| Very large dataset with nonlinear boundaries | **Consider other models** (random forests, gradient boosting, neural networks) |

---

## Common Pitfalls

### 1. Forgetting to Scale Features

SVMs are distance-based. If one feature ranges from 0 to 1000 and another from 0 to 1, the first feature will dominate the margin computation. **Always standardize or normalize your features.**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Always use a pipeline to ensure scaling
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])
model.fit(X_train, y_train)
```

### 2. Using Kernel SVM on Large Datasets

Kernel SVMs (`SVC`) have $O(m^2)$ to $O(m^3)$ training time. On a dataset with 1 million instances, training can take **days or never finish**. Use `LinearSVC` or `SGDClassifier` instead.

### 3. Not Tuning $C$ and $\gamma$

Default hyperparameters rarely give optimal results. Always use cross-validation to tune them:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

### 4. Confusing $C$ Direction with Regularization

In most models (Ridge, Lasso), $\alpha$ increases = more regularization. In SVMs, $C$ increases = **less** regularization. This is because $C$ is the penalty for violations, not the regularization strength.

$$C = \frac{1}{\alpha} \quad \Rightarrow \quad \text{large } C = \text{small regularization}$$

### 5. Using `probability=True` Unnecessarily

Enabling `probability=True` in `SVC` fits an additional Platt scaling model via cross-validation after training, which **significantly slows down** training. Only enable it if you actually need probability estimates.

### 6. Ignoring Convergence Warnings

`LinearSVC` and `SGDClassifier` may not converge with default `max_iter`. If you see convergence warnings, increase `max_iter`:

```python
# Fix convergence warning
model = LinearSVC(C=1.0, max_iter=10000)  # default is 1000
```

### 7. Choosing the Wrong Kernel

- **Start with RBF** — it's the most versatile default
- **Switch to linear** if the dataset is large or high-dimensional (text, genomics)
- **Use polynomial** only if you have domain knowledge about the degree
- **Avoid sigmoid** — it's not a valid kernel for all parameter values

### 8. Not Handling Class Imbalance

SVMs are sensitive to imbalanced classes. Use `class_weight='balanced'` to adjust:

```python
model = SVC(kernel='rbf', C=1.0, class_weight='balanced')
```

---

## Quick Reference

```python
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- CLASSIFICATION ----

# Linear SVM (large datasets)
clf = Pipeline([('scaler', StandardScaler()), ('svm', LinearSVC(C=1.0, max_iter=10000))])

# Nonlinear SVM with RBF kernel (small/medium datasets)
clf = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))])

# Polynomial kernel
clf = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='poly', degree=3, coef0=1, C=5))])

# Online SVM (streaming / very large data)
clf = Pipeline([('scaler', StandardScaler()), ('svm', SGDClassifier(loss='hinge', alpha=0.001))])

# ---- REGRESSION ----

# Linear SVR (large datasets)
reg = Pipeline([('scaler', StandardScaler()), ('svr', LinearSVR(epsilon=0.1, C=1.0, max_iter=10000))])

# Nonlinear SVR with RBF kernel
reg = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))])
```
