# SVM Questions & Answers

---

## 1. What is the fundamental idea behind Support Vector Machines?

The fundamental idea behind SVMs is to find the **optimal hyperplane** that separates two classes by **maximizing the margin** — the distance between the decision boundary and the closest training instances from each class.

Why maximize the margin? A larger margin means the model is less sensitive to small perturbations in the data, which leads to better **generalization** to unseen instances. Among all possible hyperplanes that correctly separate the classes, the maximum-margin hyperplane has the strongest theoretical guarantee of low generalization error.

Formally, given training data $\{(\mathbf{x}_i, y_i)\}$ where $y_i \in \{-1, +1\}$, the SVM finds $\mathbf{w}$ and $b$ that:

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i$$

This minimizes $\|\mathbf{w}\|$ (which maximizes the margin $\frac{2}{\|\mathbf{w}\|}$) while ensuring all instances are correctly classified.

For non-separable data, the idea extends to **soft margin** SVMs that tolerate some misclassifications (controlled by $C$), and for nonlinear problems, the **kernel trick** allows finding a maximum-margin separator in a higher-dimensional feature space without explicitly computing the transformation.

---

## 2. What is a support vector?

A **support vector** is a training instance that lies **on the margin boundary** or **inside the margin** (or on the wrong side in the soft margin case). These are the instances that directly influence the position and orientation of the decision boundary.

Mathematically, support vectors are the instances for which the Lagrange multiplier $\alpha_i > 0$ in the dual formulation. They satisfy:

$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \leq 1$$

(they are either exactly on the margin, inside the margin, or misclassified).

**Key properties:**
- **The decision boundary depends only on the support vectors.** If you removed all other training instances and retrained, you would get the exact same model.
- After training, `model.support_vectors_` gives you the support vectors and `model.n_support_` tells you how many there are per class.
- Fewer support vectors generally indicates a simpler, more efficient model with better generalization.
- At prediction time, only the support vectors are used, not the full training set.

```python
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)

print(f"Number of support vectors per class: {model.n_support_}")
print(f"Total support vectors: {len(model.support_vectors_)}")
print(f"Support vector indices: {model.support_}")
```

---

## 3. Why is it important to scale the inputs when using SVMs?

SVMs are **distance-based** algorithms. The margin, the kernel computations (e.g., RBF computes $\|\mathbf{x}_i - \mathbf{x}_j\|^2$), and the regularization term $\|\mathbf{w}\|^2$ all depend on the **magnitude of features**.

**If features are on different scales, the SVM will:**
- Give disproportionate importance to features with larger ranges
- Compute distorted distances, leading to a suboptimal margin
- Converge slowly or fail to converge

**Example:** Consider two features — "age" (range 18–80) and "income" (range 20,000–200,000). Without scaling, the distance between two points is dominated almost entirely by the income dimension. The SVM essentially ignores age.

**The fix is simple — always standardize:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),  # zero mean, unit variance
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])
```

This applies to all SVM variants: `SVC`, `LinearSVC`, `SVR`, and `SGDClassifier` with hinge loss.

> **Note:** `gamma='scale'` (default in scikit-learn) sets $\gamma = \frac{1}{n \cdot \text{Var}(X)}$, which partially accounts for feature scale. However, scaling your features explicitly is still recommended for best results and convergence speed.

---

## 4. Can an SVM classifier output a confidence score when it classifies an instance? What about a probability?

**Confidence score — Yes.** The SVM's decision function $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$ gives a signed distance from the decision boundary. The **magnitude** of this value can be interpreted as confidence:
- $|f(\mathbf{x})|$ large → the instance is far from the boundary → high confidence
- $|f(\mathbf{x})|$ small → the instance is near the boundary → low confidence

```python
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)

# Confidence scores (signed distance from boundary)
scores = model.decision_function(X_test)
print(scores)  # e.g., [-2.3, 0.1, 4.7] — negative = class 0, positive = class 1
```

**Probability — Yes, but with caveats.** `SVC` can output probability estimates using **Platt scaling** (fitting a logistic regression on the SVM scores via internal cross-validation). You enable this by setting `probability=True`:

```python
model = SVC(kernel='rbf', C=1.0, probability=True)
model.fit(X_train, y_train)

# Probability estimates
probs = model.predict_proba(X_test)
print(probs)  # e.g., [[0.85, 0.15], [0.47, 0.53]]
```

**Caveats of `probability=True`:**
- **Slows down training** significantly (requires an additional 5-fold cross-validation internally)
- **Probabilities may be poorly calibrated** — Platt scaling assumes the scores follow a specific distribution, which may not hold
- The probabilities and the `decision_function` scores may be **inconsistent** — an instance with a higher confidence score doesn't always get a higher probability from Platt scaling

`LinearSVC` does not have `predict_proba()`. If you need probabilities with a linear SVM, you can wrap it with `CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

model = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000))
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)
```

---

## 5. Should you use the primal or the dual form of the SVM problem to train a model on a training set with millions of instances and hundreds of features?

**Use the primal form.**

The choice depends on the relationship between $m$ (instances) and $n$ (features):

| Form | Number of variables | Better when |
|------|-------------------|-------------|
| **Primal** | $O(n)$ — one per feature | $m \gg n$ (many instances, few features) |
| **Dual** | $O(m)$ — one per instance | $n \gg m$ (many features, few instances) |

With millions of instances and hundreds of features, $m \gg n$. The dual form would have **millions** of variables (one $\alpha_i$ per instance) and would need to compute the $m \times m$ kernel matrix, which is computationally prohibitive.

The primal form has only hundreds of variables (one per feature + bias), making it far more efficient.

**In practice:** Use `LinearSVC` (which uses the liblinear solver that works on the primal form) or `SGDClassifier(loss='hinge')` (which uses stochastic gradient descent on the primal formulation).

```python
# Primal solvers for large datasets
from sklearn.svm import LinearSVC
model = LinearSVC(C=1.0, max_iter=10000)  # liblinear, primal

from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='hinge', alpha=0.001, max_iter=1000)  # SGD, primal
```

> **Important:** The dual form is necessary for the **kernel trick**. If you need a nonlinear SVM, you must use the dual form (`SVC`). But with millions of instances, kernel SVMs are generally impractical anyway — consider other nonlinear models (random forests, gradient boosting, neural networks).

---

## 6. Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease $\gamma$? What about $C$?

**If the model is underfitting, you should increase both $\gamma$ and $C$.**

Here's why:

**$\gamma$ (gamma):**
- $\gamma$ controls how much influence each training instance has
- **Small $\gamma$** → each instance has a wide influence → the decision boundary is **smooth and simple** → potentially underfitting
- **Large $\gamma$** → each instance has a narrow influence → the decision boundary becomes **more complex and wiggly** → can capture more detail
- **Increase $\gamma$** to allow the model to learn more complex patterns

The RBF kernel is $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$. A larger $\gamma$ makes the Gaussian bell curve narrower, so only very close points influence each other.

**$C$ (regularization):**
- $C$ controls how much the model penalizes margin violations
- **Small $C$** → many violations allowed → wider margin → **more regularization** → potentially underfitting
- **Large $C$** → fewer violations allowed → narrower margin → **less regularization** → fits training data more tightly
- **Increase $C$** to reduce regularization and let the model fit the training data better

**Summary:**

| Hyperparameter | Underfitting fix | Overfitting fix |
|---------------|-----------------|-----------------|
| $\gamma$ | Increase ↑ | Decrease ↓ |
| $C$ | Increase ↑ | Decrease ↓ |

> **Caution:** Don't increase both too aggressively at once, or you'll swing from underfitting to overfitting. Use **grid search with cross-validation** to find the right balance.

---

## 7. How should you set the QP parameters ($H$, $f$, $A$, and $b$) to solve the soft margin linear SVM classifier problem using an off-the-shelf QP solver?

The QP solver minimizes:

$$\min_{\mathbf{p}} \quad \frac{1}{2}\mathbf{p}^T H \mathbf{p} + \mathbf{f}^T \mathbf{p} \quad \text{subject to} \quad A\mathbf{p} \leq \mathbf{b}$$

For a soft margin linear SVM, we need to map the SVM formulation to this QP form. The SVM problem is:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{m}\xi_i$$

Subject to: $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i$ and $\xi_i \geq 0$ for all $i$.

**The QP variable vector $\mathbf{p}$:**

$$\mathbf{p} = \begin{bmatrix} \mathbf{w} \\ b \\ \boldsymbol{\xi} \end{bmatrix} \quad \text{(size: } n + 1 + m \text{)}$$

**$H$ (Hessian matrix) — quadratic part:**

$$H = \begin{bmatrix} I_n & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \quad \text{(size: } (n+1+m) \times (n+1+m) \text{)}$$

The identity matrix $I_n$ corresponds to $\|\mathbf{w}\|^2$. The bias $b$ and slack variables $\xi_i$ have no quadratic terms.

**$\mathbf{f}$ (linear part):**

$$\mathbf{f} = \begin{bmatrix} \mathbf{0}_n \\ 0 \\ C \cdot \mathbf{1}_m \end{bmatrix}$$

Zero for $\mathbf{w}$ and $b$; $C$ for each slack variable (from the $C\sum\xi_i$ term).

**$A$ and $\mathbf{b}$ (inequality constraints):**

We have two sets of constraints to convert to $A\mathbf{p} \leq \mathbf{b}$:

1. **Classification constraints:** $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i$ → rewritten as $-(y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 + \xi_i) \leq 0$

2. **Non-negativity of slack:** $\xi_i \geq 0$ → rewritten as $-\xi_i \leq 0$

$$A = \begin{bmatrix} -y_1 \mathbf{x}_1^T & -y_1 & -e_1^T \\ \vdots & \vdots & \vdots \\ -y_m \mathbf{x}_m^T & -y_m & -e_m^T \\ 0 & 0 & -I_m \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} -\mathbf{1}_m \\ \mathbf{0}_m \end{bmatrix}$$

Where $e_i$ is the $i$-th standard basis vector in $\mathbb{R}^m$ (picking out the $i$-th slack variable).

---

## 8. Train a LinearSVC on a linearly separable dataset. Then train an SVC and a SGDClassifier on the same dataset. See if you can get them to produce roughly the same model.

```python
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 1. Create a linearly separable dataset
X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=3.0, random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train LinearSVC
linear_svc = LinearSVC(C=1.0, loss='hinge', max_iter=10000, random_state=42)
linear_svc.fit(X_scaled, y)

# 3. Train SVC with linear kernel
svc = SVC(kernel='linear', C=1.0, random_state=42)
svc.fit(X_scaled, y)

# 4. Train SGDClassifier with hinge loss
# Key: match regularization — SGD's alpha ≈ 1 / (n_samples * C)
m = len(X_scaled)
C = 1.0
sgd = SGDClassifier(
    loss='hinge',
    alpha=1 / (m * C),  # match C from SVM
    max_iter=10000,
    tol=1e-4,
    random_state=42
)
sgd.fit(X_scaled, y)

# 5. Compare the models
print("=== Model Comparison ===\n")
print(f"{'Model':<20} {'w1':>10} {'w2':>10} {'bias':>10}")
print("-" * 52)
print(f"{'LinearSVC':<20} {linear_svc.coef_[0][0]:>10.4f} {linear_svc.coef_[0][1]:>10.4f} {linear_svc.intercept_[0]:>10.4f}")
print(f"{'SVC (linear)':<20} {svc.coef_[0][0]:>10.4f} {svc.coef_[0][1]:>10.4f} {svc.intercept_[0]:>10.4f}")
print(f"{'SGDClassifier':<20} {sgd.coef_[0][0]:>10.4f} {sgd.coef_[0][1]:>10.4f} {sgd.intercept_[0]:>10.4f}")

# 6. Visualize the decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models = [('LinearSVC', linear_svc), ('SVC (linear)', svc), ('SGDClassifier', sgd)]

for ax, (name, model) in zip(axes, models):
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='RdBu', alpha=0.6, edgecolors='k', s=20)

    # Plot decision boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    XX, YY = np.meshgrid(xx, yy)
    Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

    ax.contour(XX, YY, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
               linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    ax.set_title(name)

plt.suptitle('Decision Boundaries Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

**Key points to get similar models:**
- **Same $C$ value** for `LinearSVC` and `SVC`
- **Match regularization for SGDClassifier:** set `alpha = 1 / (n_samples * C)` because SGD uses $\alpha\|\mathbf{w}\|^2/2$ while SVM uses $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum\text{hinge}$
- **Scale the features** — all three models benefit from standardization
- **Use `loss='hinge'`** for both `LinearSVC` and `SGDClassifier`
- The weights and biases should be very close but not identical (different solvers: liblinear vs libsvm vs SGD)

---

## 9. Train an SVM classifier on the MNIST dataset. Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all 10 digits. You may want to tune the hyperparameters using small validation sets to speed up the process. What accuracy can you reach?

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, loguniform

# 1. Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2. Split into train/test (MNIST standard: 60k train, 10k test)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 3. Scale the features (critical for SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.transform(X_test.astype(np.float64))

# 4. Quick baseline with default RBF kernel on a subset
# (Training SVC on 60k samples is slow — use a subset for tuning)
subset_size = 10000
X_sub = X_train_scaled[:subset_size]
y_sub = y_train[:subset_size]

# 5. Hyperparameter tuning on the subset
param_distributions = {
    'C': loguniform(0.1, 100),           # log-uniform between 0.1 and 100
    'gamma': loguniform(1e-4, 1e-1),     # log-uniform between 0.0001 and 0.1
}

svm = SVC(kernel='rbf', decision_function_shape='ovr')  # one-vs-rest for 10 digits

random_search = RandomizedSearchCV(
    svm, param_distributions,
    n_iter=20,      # try 20 random combinations
    cv=3,           # 3-fold CV on the subset
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_sub, y_sub)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV accuracy (on subset): {random_search.best_score_:.4f}")

# 6. Train on the FULL training set with the best hyperparameters
best_model = SVC(kernel='rbf', **random_search.best_params_)
best_model.fit(X_train_scaled, y_train)

# 7. Evaluate on test set
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy on MNIST: {test_accuracy:.4f}")
```

**Expected results:**
- With a well-tuned RBF kernel (typically $C \approx 5\text{-}10$, $\gamma \approx 0.01\text{-}0.03$), you can reach approximately **97-98% accuracy** on MNIST.
- Training `SVC` on the full 60,000 samples takes significant time due to $O(m^2)$ complexity.

**Notes:**
- `SVC` uses **one-vs-one (OvO)** by default for multi-class (trains $\binom{10}{2} = 45$ classifiers). Setting `decision_function_shape='ovr'` changes the output format but the underlying training is still OvO.
- To use true **one-vs-all (OvR)**, wrap with `OneVsRestClassifier`:

```python
from sklearn.multiclass import OneVsRestClassifier
ovr_svm = OneVsRestClassifier(SVC(kernel='rbf', C=5, gamma=0.02))
ovr_svm.fit(X_train_scaled, y_train)
```

- For **faster** alternatives: `LinearSVC` with OvR can also reach ~92-94% and trains much faster.

---

## 10. Train an SVM regressor on the California housing dataset.

```python
import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

# 1. Load California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target range: {y.min():.2f} to {y.max():.2f} (median house value in $100k)")

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# Approach 1: LinearSVR (fast, for baseline)
# =========================================================
linear_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', LinearSVR(epsilon=0.1, C=1.0, max_iter=10000, random_state=42))
])
linear_svr.fit(X_train, y_train)
y_pred_linear = linear_svr.predict(X_test)

print("\n=== LinearSVR (Baseline) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_linear)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_linear):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_linear):.4f}")

# =========================================================
# Approach 2: SVR with RBF kernel (nonlinear, tuned)
# =========================================================

# Tune on a subset for speed (California housing has ~20k samples)
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

param_distributions = {
    'svr__C': loguniform(0.1, 1000),
    'svr__gamma': loguniform(1e-4, 1),
    'svr__epsilon': loguniform(0.01, 1),
}

random_search = RandomizedSearchCV(
    svr_pipeline,
    param_distributions,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)

print(f"\nBest params: {random_search.best_params_}")

# Evaluate best model
y_pred_rbf = random_search.best_estimator_.predict(X_test)

print("\n=== SVR with RBF kernel (Tuned) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rbf)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_rbf):.4f}")
print(f"R²:   {r2_score(y_test, y_pred_rbf):.4f}")

# =========================================================
# Compare models
# =========================================================
print("\n=== Comparison ===")
print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 57)
for name, y_p in [('LinearSVR', y_pred_linear), ('SVR (RBF, tuned)', y_pred_rbf)]:
    rmse = np.sqrt(mean_squared_error(y_test, y_p))
    mae = mean_absolute_error(y_test, y_p)
    r2 = r2_score(y_test, y_p)
    print(f"{name:<25} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
```

**Expected results:**
- **LinearSVR:** R² ≈ 0.55–0.60, RMSE ≈ 0.75–0.80 (limited by linearity assumption)
- **SVR with tuned RBF kernel:** R² ≈ 0.70–0.83, RMSE ≈ 0.50–0.65 (captures nonlinear relationships)

**Key takeaways:**
- **Scaling is essential** — the California housing features have very different ranges (latitude/longitude vs. income vs. number of rooms)
- **LinearSVR** is fast but underperforms because the relationship between features and house prices is nonlinear
- **SVR with RBF kernel** captures nonlinearity and achieves significantly better results
- **Tuning $C$, $\gamma$, and $\varepsilon$** is critical — default values typically underperform
- SVR on ~20k samples with RBF kernel is feasible; for datasets much larger than this, consider `LinearSVR` or other nonlinear models (gradient boosting, random forests)
