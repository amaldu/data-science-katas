# Logistic Regression. End-to-End Implementation Guide

> Implementation guide for Logistic Regression. Theory and background in [cheatsheet_logistic_regression.md](cheatsheet_logistic_regression.md).

---

## Step 0 -- Define the Problem

Before writing any code, confirm two things:

1. **The target variable is categorical (binary).** If it is continuous, you need regression (e.g., [linear regression](linear_regression/step_by_step_linear_regression.md)), not classification.
2. **Decide what matters more: interpretability or raw predictive power.**

| Goal | Model direction |
|:---|:---|
| Understand which features drive the outcome, report odds ratios, statistical significance | Stay with logistic regression |
| Maximize classification accuracy, don't need to explain individual coefficients | Consider tree-based models, SVMs, or neural networks later if logistic regression underperforms |

---

## Step 1 -- Explore and Clean the Data (EDA)

EDA has one purpose: understand the data well enough to make informed preprocessing decisions. Do these checks in order.

### 1.1 Target Distribution (Class Balance)

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class_counts = pd.Series(y).value_counts()
print(f"Class distribution:\n{class_counts}")
print(f"Class ratio (minority/majority): {class_counts.min() / class_counts.max():.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
class_counts.plot(kind='bar', edgecolor='k', alpha=0.8, ax=ax)
ax.set_title('Target Class Distribution')
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

| Observation | Action |
|:---|:---|
| Roughly balanced (ratio > 0.5) | Use data as-is |
| Moderate imbalance (ratio 0.2-0.5) | Use `class_weight='balanced'` in `LogisticRegression`, or oversample minority with SMOTE |
| Severe imbalance (ratio < 0.2) | Combine `class_weight='balanced'` with adjusted threshold; consider SMOTE, undersampling, or cost-sensitive learning. **Do not use accuracy as your primary metric** -- use PR-AUC or F1 instead |

### 1.2 Missing Values

```python
missing = X.isnull().sum()
missing_pct = (missing / len(X)) * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))
```

| Missing % | Strategy |
|:---|:---|
| < 5% | Impute with median (numeric) or mode (categorical) |
| 5-30% | Model-based imputation (e.g., `IterativeImputer`) or create a "missing" indicator feature |
| > 30% | Consider dropping the feature entirely |

### 1.3 Outlier Detection

```python
from scipy import stats

z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
outlier_rows = (z_scores > 3).any(axis=1)
print(f"Outlier rows (Z > 3): {outlier_rows.sum()} / {len(X)}")
```

| Situation | Action |
|:---|:---|
| Few outliers (< 1% of data), likely data errors | Remove them |
| Moderate outliers, real but extreme values | Cap at 1st/99th percentile (winsorize) |
| Many outliers, or outliers carry meaningful signal | Keep them, use robust methods (RobustScaler) |

### 1.4 Feature-Target Relationships

For classification, examine how features relate to the binary outcome:

```python
import seaborn as sns

numeric_cols = X.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Box plot by class
    for cls in [0, 1]:
        axes[0].boxplot(X.loc[y == cls, col], positions=[cls], widths=0.6)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Class 0', 'Class 1'])
    axes[0].set_title(f'{col} by Class')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram by class
    axes[1].hist(X.loc[y == 0, col], bins=30, alpha=0.6, label='Class 0', edgecolor='k')
    axes[1].hist(X.loc[y == 1, col], bins=30, alpha=0.6, label='Class 1', edgecolor='k')
    axes[1].set_title(f'{col} Distribution by Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

Look for:
- **Clear separation** between classes -- strong predictive feature.
- **Overlapping distributions** -- weak feature on its own, but may still contribute in combination.
- **Non-linear separation** -- flag for polynomial or interaction features in Step 3.

### 1.5 Encode Categorical Features

| Feature type | Encoding |
|:---|:---|
| Nominal (no order): color, city | One-hot encoding (`pd.get_dummies` or `OneHotEncoder`) |
| Ordinal (has order): low/medium/high | Ordinal encoding (map to integers) |
| High cardinality (>20 categories) | Target encoding or frequency encoding |

```python
X = pd.get_dummies(X, columns=['category_col'], drop_first=True)
```

---

## Step 2 -- Split the Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Critical rule:** Everything from this point forward (scaling, feature engineering, model fitting) is done on `X_train` / `y_train` only. The test set is touched only once, at the very end, for final evaluation. Fitting a scaler or imputer on the full dataset before splitting causes **data leakage**.

**Important for classification:** Always use `stratify=y` to preserve the class distribution across train and test sets, especially when classes are imbalanced.

| Dataset size | Split ratio |
|:---|:---|
| < 1,000 samples | 80/20 and use cross-validation heavily |
| 1,000 - 100,000 | 80/20 is standard |
| > 100,000 | 90/10 or even 95/5 (test set is already large enough) |

---

## Step 3 -- Feature Engineering and Scaling

### 3.1 Multicollinearity Check (VIF)

> See [Multicollinearity](logistic_regression_cheatsheet.md#3-little-or-no-multicollinearity) in the cheatsheet for full theory.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_data = pd.DataFrame()
vif_data['Feature'] = X_train.columns
vif_data['VIF'] = [
    variance_inflation_factor(X_train.values, i)
    for i in range(X_train.shape[1])
]
print(vif_data.sort_values('VIF', ascending=False))
```

| VIF result | Action |
|:---|:---|
| All VIF < 5 | Proceed -- no multicollinearity issue |
| VIF 5-10 for some features | Monitor. Consider combining correlated features or dropping one |
| VIF > 10 | **Must act.** Drop one feature from each correlated pair, apply PCA, or use L2 regularization (which handles multicollinearity natively) |

### 3.2 Feature Scaling

> See [Feature Scaling](../07_feature_engineering/feature_scaling.md) for the full guide on each method.

Logistic regression is solved iteratively via gradient-based optimization. **Scaling is strongly recommended** -- unscaled features cause slow convergence and can bias regularization toward features with larger magnitudes.

**Default choice: StandardScaler** (Z-score normalization). Use RobustScaler if you kept outliers.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use train's μ and σ
```

### 3.3 Polynomial / Interaction Features (if needed)

Only add these if EDA (Step 1.4) revealed non-linear separation boundaries between classes.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
```

| Situation | Approach |
|:---|:---|
| Linear decision boundary appears sufficient | Skip polynomial features |
| Mild non-linearity in one or two features | degree=2, consider `interaction_only=True` |
| Complex non-linear boundary | degree=2 or 3, but **pair with strong regularization** to prevent overfitting |

**After adding polynomial features, re-check VIF** -- polynomial terms are often correlated. If VIF explodes, pair with L2 regularization.

---

## Step 4 -- Fit the Model

### 4.1 Choose the Solver

> See [Gradient Descent for Logistic Regression](logistic_regression_cheatsheet.md#gradient-descent-for-logistic-regression) in the cheatsheet.

Scikit-learn's `LogisticRegression` supports multiple solvers. The choice depends on dataset size and regularization:

| Scenario | Solver | Regularization support |
|:---|:---|:---|
| Small to medium dataset (< 10,000 samples) | `lbfgs` (default) | L2 only |
| Large dataset or sparse features | `saga` | L1, L2, Elastic Net |
| Need L1 regularization (feature selection) | `liblinear` or `saga` | L1, L2 |
| Multiclass (multinomial) | `lbfgs` or `saga` | L2 / L1+L2 |

### 4.2 Fit the Baseline

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
```

Record training performance as a baseline:

```python
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training ROC-AUC:  {train_auc:.4f}")
print(f"\n{classification_report(y_train, y_train_pred)}")
```

---

## Step 5 -- Validate Assumptions

Logistic regression has fewer distributional assumptions than linear regression (no normality, homoscedasticity, or linearity-in-$y$ requirements), but the assumptions it does have are critical.

> See [Key Assumptions](logistic_regression_cheatsheet.md#key-assumptions) in the cheatsheet for the full theory behind each assumption.

### Assumption Validation Decision Tree

```
                          ┌─────────────┐
                          │  Fit Model   │
                          └──────┬───────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  1. Binary/Ordinal        │
                    │     Outcome?              │
                    └───┬──────────────────┬────┘
                   Pass │                  │ Fail: target is multi-class nominal
                        ▼                  ▼
                  ┌──────────┐   Use Softmax (multinomial) or OvR
                  │          │
                  ▼
            ┌──────────────────────────┐
            │  2. Independence of      │
            │     Observations?        │
            └───┬──────────────────┬───┘
           Pass │                  │ Fail: repeated measures / time series
                ▼                  ▼
          ┌──────────┐   Use mixed-effects logistic regression
          │          │   or GEE
          ▼
    ┌──────────────────────────┐
    │  3. No Multicollinearity? │
    │     (VIF check)           │
    └───┬──────────────────┬────┘
   Pass │                  │ Fail: VIF > 10
        ▼                  ▼
  ┌──────────┐   Drop features, PCA, or
  │          │   use L2 regularization → refit
  ▼
┌──────────────────────────┐
│  4. Linear Log-Odds?     │
│  (Box-Tidwell or         │
│   visual check)          │
└───┬──────────────────┬───┘
   Pass │              │ Fail: curved relationship
        ▼              ▼
  ┌──────────┐   Add polynomial/interaction terms
  │          │   or transform features → refit
  ▼
┌──────────────────────────┐
│  5. Sufficient Sample    │
│     Size?                │
│  (≥ 10-20 EPV)          │
└───┬──────────────────┬───┘
   Pass │              │ Fail: too few events per variable
        ▼              ▼
  Proceed to       Reduce features, use regularization,
  Step 6           or collect more data
```

### 5.1 Check Linear Relationship with Log-Odds

This is the most important assumption unique to logistic regression. The relationship between each continuous feature and the **log-odds** of the outcome must be approximately linear.

```python
import statsmodels.api as sm

X_with_const = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_with_const).fit(disp=0)

feature_names = X_train.columns if hasattr(X_train, 'columns') else [
    f'x{i}' for i in range(X_train_scaled.shape[1])
]

for i, col in enumerate(feature_names):
    feature_vals = X_train_scaled[:, i]
    n_bins = 10
    bins = pd.qcut(feature_vals, q=n_bins, duplicates='drop')
    
    df_temp = pd.DataFrame({'feature': feature_vals, 'target': y_train, 'bin': bins})
    grouped = df_temp.groupby('bin')['target'].mean()
    log_odds = np.log(grouped / (1 - grouped + 1e-10))
    bin_centers = df_temp.groupby('bin')['feature'].mean()
    
    plt.figure(figsize=(8, 4))
    plt.scatter(bin_centers, log_odds, edgecolors='k')
    plt.xlabel(f'{col} (binned means)')
    plt.ylabel('Empirical Log-Odds')
    plt.title(f'Log-Odds Linearity Check: {col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

| What you see | Diagnosis | Next step |
|:---|:---|:---|
| Roughly linear trend | Linearity in log-odds holds | Move to 5.2 |
| U-shape or arch | Non-linear log-odds relationship | Add $x^2$ for the offending feature (Step 3.3), refit |
| S-shape or irregular | Complex non-linearity | Transform feature ($\log$, $\sqrt{}$) or switch to a more flexible model |

### 5.2 Check Independence

This is most relevant when data has a natural ordering (time series, repeated measures on same subjects).

For cross-sectional data with independent subjects, this assumption is usually safe to assume. For repeated measures or clustered data, use mixed-effects logistic regression or GEE.

### 5.3 Check Multicollinearity

Already covered in Step 3.1 (VIF check). If VIF > 10 for any feature after fitting the baseline, go back to Step 3 and address it.

### 5.4 Check Sample Size (Events Per Variable)

```python
n_features = X_train_scaled.shape[1]
n_minority = min(np.bincount(y_train))
epv = n_minority / n_features

print(f"Number of features:       {n_features}")
print(f"Minority class count:     {n_minority}")
print(f"Events Per Variable (EPV): {epv:.1f}")
```

| EPV | Diagnosis | Next step |
|:---|:---|:---|
| ≥ 20 | Sufficient sample size | Proceed to Step 6 |
| 10-20 | Borderline | Use regularization, interpret with caution |
| < 10 | Insufficient | Reduce features (Lasso, PCA), use penalized MLE (Firth's method), or collect more data |

### Assumptions Summary: Quick Reference

| Assumption | Diagnostic tool | Pass criterion | What breaks if violated |
|:---|:---|:---|:---|
| Binary outcome | Inspect target | Two classes | Model is fundamentally wrong |
| Independence | Study design review | Independent observations | Inflated significance (false positives) |
| No multicollinearity | VIF | All VIF < 10 | Unstable, uninterpretable coefficients |
| Linear log-odds | Binned log-odds plot | Approximately linear trend | Biased coefficients, poor predictions |
| Large sample size | EPV calculation | EPV ≥ 10-20 | Extreme estimates, non-convergence |

---

## Step 6 -- Evaluate Performance

> See [Classification Evaluation Metrics](../06_metrics/classification_metrics_cheatsheet.md) for detailed formulas, from-scratch code, and when to use each metric.

### 6.1 Test Set Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Test Accuracy:   {test_accuracy:.4f}")
print(f"Test Precision:  {test_precision:.4f}")
print(f"Test Recall:     {test_recall:.4f}")
print(f"Test F1:         {test_f1:.4f}")
print(f"Test ROC-AUC:    {test_auc:.4f}")

print(f"\n{classification_report(y_test, y_test_pred)}")

ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
plt.title('Confusion Matrix')
plt.show()
```

### 6.2 ROC and Precision-Recall Curves

```python
from sklearn.metrics import roc_curve, precision_recall_curve, auc

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0].plot(fpr, tpr, lw=2, label=f'ROC (AUC = {test_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Precision-Recall curve
precisions, recalls, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = auc(recalls, precisions)
axes[1].plot(recalls, precisions, lw=2, label=f'PR (AUC = {pr_auc:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 6.3 Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
cv_auc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')

print(f"CV Accuracy: {cv_accuracy.mean():.4f} +/- {cv_accuracy.std():.4f}")
print(f"CV ROC-AUC:  {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
print(f"CV F1:       {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
```

**Use `StratifiedKFold`** instead of regular `KFold` to preserve class distribution in each fold, especially with imbalanced data.

### 6.4 Diagnose the Results

Compare training metrics (Step 4.2) with test/CV metrics and use this decision table:

| Training AUC | Test AUC | Gap | CV std | Diagnosis | Action |
|:---|:---|:---|:---|:---|:---|
| < 0.6 | < 0.6 | Small | Low | **Underfitting** | Add features, add polynomial terms, reduce regularization |
| > 0.95 | < 0.75 | Large (> 0.15) | High | **Overfitting** | Add regularization (Step 7), reduce features, get more data |
| > 0.75 | > 0.75 | Small (< 0.05) | Low | **Good fit** | Proceed to Step 8 |
| > 0.75 | 0.6-0.75 | Moderate | Moderate | **Mild overfitting** | Try light regularization, or remove weakest features |
| Any | Any | Any | > 0.1 * mean | **Unstable model** | Get more data, simplify the model, or increase $k$ in CV |

**Underfitting remedies** (go back to earlier steps):
- Add more relevant features → Step 1 / Step 3
- Add polynomial or interaction terms → Step 3.3
- Reduce regularization strength if already applied → Step 7
- Consider a non-linear model (trees, ensembles) → outside this guide

**Overfitting remedies** (proceed to Step 7):
- Apply L1, L2, or Elastic Net regularization → Step 7
- Remove irrelevant features (use L1 for automatic selection)
- Collect more training data

---

## Step 7 -- Regularization and Threshold Tuning

### 7.1 Regularization

Regularization is triggered by **overfitting** (Step 6) or **multicollinearity** (Step 3.1). It is not always needed -- skip this if your baseline model from Step 4 already performs well.

> See [Regularization Techniques](../07_feature_engineering/regularization_techniques.md) for full theory on Ridge, Lasso, and Elastic Net.
> See [Regularization for Logistic Regression](logistic_regression_cheatsheet.md#regularization) in the cheatsheet.

Note: In scikit-learn's `LogisticRegression`, the regularization parameter is `C` (inverse of $\lambda$). **Smaller `C` = stronger regularization.**

| Situation | Regularization (`penalty`) | Why |
|:---|:---|:---|
| All features believed relevant, multicollinearity present | **L2 (`penalty='l2'`)** | Shrinks all coefficients, never removes features, stabilizes correlated coefficients |
| Many features suspected irrelevant | **L1 (`penalty='l1'`)** | Drives irrelevant coefficients to exactly zero (automatic feature selection) |
| Correlated groups + some irrelevant features | **Elastic Net (`penalty='elasticnet'`)** | Combines L2's group stability with L1's sparsity |
| Not sure which to use | **L2** with moderate `C` | Safe default for logistic regression |

### 7.2 Tune Hyperparameters via Cross-Validation

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l1', 'l2'],
    'solver': ['saga'],
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=2000),
    param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

if best_model.penalty == 'l1':
    n_selected = np.sum(best_model.coef_ != 0)
    print(f"L1 features selected: {n_selected} / {X_train_scaled.shape[1]}")
```

### 7.3 Threshold Tuning

> See [Decision Rule and Threshold Selection](logistic_regression_cheatsheet.md#decision-rule) in the cheatsheet for full theory.

The default threshold of 0.5 is rarely optimal. Tune it based on your business objective:

```python
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score

y_val_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Method 1: Maximize F1
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_val_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_threshold_f1 = thresholds_pr[best_f1_idx]
print(f"Best F1 threshold: {best_threshold_f1:.3f} (F1 = {f1_scores[best_f1_idx]:.3f})")

# Method 2: Youden's J (maximize TPR - FPR)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_val_proba)
j_scores = tpr - fpr
best_j_idx = np.argmax(j_scores)
best_threshold_j = thresholds_roc[best_j_idx]
print(f"Youden's J threshold: {best_threshold_j:.3f}")
```

| Business need | Threshold strategy |
|:---|:---|
| Minimize false negatives (disease screening, fraud) | **Lower threshold** (e.g., 0.2-0.4) to maximize recall |
| Minimize false positives (spam filter, criminal conviction) | **Raise threshold** (e.g., 0.6-0.8) to maximize precision |
| Balance precision and recall | Optimize for **F1 score** |
| General-purpose | Use **Youden's J statistic** |

### 7.4 Evaluate the Tuned Model

After fitting the regularized model and selecting a threshold, repeat Step 6 metrics and compare against the baseline:

```python
for name, m, threshold in [
    ("Baseline (0.5)", model, 0.5),
    ("Tuned (best F1)", best_model, best_threshold_f1),
    ("Tuned (Youden)", best_model, best_threshold_j),
]:
    proba = m.predict_proba(X_test_scaled)[:, 1]
    preds = (proba >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc_val = roc_auc_score(y_test, proba)
    print(f"{name:25s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc_val:.4f}")
```

| Outcome | Interpretation | Next step |
|:---|:---|:---|
| Regularized AUC > Baseline AUC on test set | Regularization helped reduce overfitting | Keep the regularized model |
| Regularized AUC ≈ Baseline AUC | Model was not overfitting much | Use whichever is simpler; L2 if multicollinearity was an issue |
| Regularized AUC < Baseline AUC | Regularization is too strong or wrong type | Try a larger `C`, or a different penalty |
| L1 set most coefficients to zero | Most features are irrelevant | Rebuild model with only the selected features for interpretability |

After choosing the best model, **re-validate assumptions** (Step 5) on the new model to confirm the remedies hold.

---

## Step 8 -- Final Model and Reporting

### 8.1 Final Test Evaluation

```python
final_model = best_model
final_threshold = best_threshold_f1  # or whichever performed best

y_final_proba = final_model.predict_proba(X_test_scaled)[:, 1]
y_final_pred = (y_final_proba >= final_threshold).astype(int)

final_accuracy = accuracy_score(y_test, y_final_pred)
final_precision = precision_score(y_test, y_final_pred)
final_recall = recall_score(y_test, y_final_pred)
final_f1 = f1_score(y_test, y_final_pred)
final_auc = roc_auc_score(y_test, y_final_proba)

print("=== Final Model Performance ===")
print(f"Threshold:   {final_threshold:.3f}")
print(f"Accuracy:    {final_accuracy:.4f}")
print(f"Precision:   {final_precision:.4f}")
print(f"Recall:      {final_recall:.4f}")
print(f"F1:          {final_f1:.4f}")
print(f"ROC-AUC:     {final_auc:.4f}")
```

### 8.2 Inspect Coefficients and Odds Ratios

```python
feature_names = X_train.columns if hasattr(X_train, 'columns') else [
    f'x{i}' for i in range(X_train_scaled.shape[1])
]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': final_model.coef_[0],
    'Odds Ratio': np.exp(final_model.coef_[0])
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df.to_string(index=False))
print(f"\nIntercept: {final_model.intercept_[0]:.4f}")
```

**Interpreting odds ratios:** An odds ratio of 2.5 for feature $x_j$ means that a one-unit increase in $x_j$ (after scaling) multiplies the odds of the positive class by 2.5, holding all other features constant.

| Odds Ratio | Interpretation |
|:---|:---|
| > 1 | Increases odds of positive class |
| = 1 | No effect |
| < 1 | Decreases odds of positive class |

### 8.3 Reporting Checklist

Document these items for reproducibility and transparency:

| Item | What to record |
|:---|:---|
| **Data** | Source, size ($m$ samples, $n$ features), date range, class distribution, any filters applied |
| **Preprocessing** | Missing value strategy, outlier handling, scaling method, encoding of categoricals |
| **Class imbalance handling** | Strategy used (class weights, SMOTE, threshold tuning, none) |
| **Assumption checks** | Which passed, which failed, what remedies were applied |
| **Model** | Solver, regularization type and strength (`C`, `penalty`), threshold used |
| **Performance** | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC on held-out test set |
| **Cross-validation** | CV metric means and standard deviations |
| **Limitations** | Any violated assumptions not fully remedied, known failure modes, class imbalance concerns |
| **Feature importance** | Top coefficients (positive and negative), their odds ratios, and magnitudes |

---

## Full Workflow Summary

```
Step 0: Define the Problem
  └─ Is target binary/categorical? → Yes → proceed. No → use regression.

Step 1: EDA
  ├─ Class imbalance?     → use class_weight, SMOTE, or adjust threshold
  ├─ Missing values?      → impute or drop
  ├─ Outliers?            → remove, cap, or keep
  ├─ Feature separation?  → identify strong/weak predictors
  └─ Categorical features?→ encode appropriately

Step 2: Train/Test Split
  └─ Use stratify=y; all fitting from here uses training set only

Step 3: Feature Engineering
  ├─ VIF check            → drop, combine, PCA, or use L2
  ├─ Scale features       → StandardScaler (strongly recommended)
  └─ Polynomial features  → only if non-linear separation flagged in Step 1

Step 4: Fit Baseline Model
  └─ LogisticRegression() → record training accuracy, AUC, F1

Step 5: Validate Assumptions
  ├─ Linear log-odds fails?    → add poly terms or transform → refit (Step 3)
  ├─ Multicollinearity?        → drop features or use L2 → refit
  ├─ Independence fails?       → use mixed-effects or GEE
  └─ Insufficient sample size? → reduce features or regularize

Step 6: Evaluate Performance
  ├─ Underfitting?  → add features/complexity (Step 3)
  ├─ Overfitting?   → regularize (Step 7)
  └─ Good fit?      → proceed to Step 7 for threshold tuning, then Step 8

Step 7: Regularization and Threshold Tuning
  ├─ All features relevant + multicollinearity → L2
  ├─ Many irrelevant features                 → L1
  ├─ Correlated groups + irrelevant features  → Elastic Net
  └─ Tune threshold based on business need (F1, Youden's J, or custom)

Step 8: Final Evaluation and Reporting
  └─ Report metrics, coefficients, odds ratios, threshold, limitations
```
