# Linear Regression. End-to-End Implementation Guide

> Implementation guide for Linear Regression. Theory and background in [cheatsheet_linear_regression_and_grad_descent.md](cheatsheet_linear_regression_and_grad_descent.md).

---

## Step 0 -- Define the Problem

Before writing any code, confirm two things:

1. **The target variable is continuous.** If it is categorical (yes/no, class labels), you need classification (e.g., [logistic regression](logistic_regression_cheatsheet.md)), not linear regression.
2. **Decide what matters more: interpretability or raw predictive power.**

| Goal | Model direction |
|:---|:---|
| Understand which features drive the outcome, report coefficients, statistical significance | Stay with linear regression |
| Maximize prediction accuracy, don't need to explain individual coefficients | Consider tree-based models or neural networks later if linear regression underperforms |

---

## Step 1 -- Explore and Clean the Data (EDA)

EDA has one purpose: understand the data well enough to make informed preprocessing decisions. Do these checks in order.

### 1.1 Target Distribution

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].hist(y, bins=40, edgecolor='k', alpha=1)
axes[0].set_title('Target Distribution')
axes[0].grid(True)
axes[1].hist(np.log1p(y), bins=40, edgecolor='k', alpha=0.7)
axes[1].set_title('Log-Transformed Target')
axes[1].grid(True)
plt.tight_layout()
plt.show()

from scipy.stats import skew, kurtosis
print(f"Skewness: {skew(y):.2f}")
print(f"Kurtosis: {kurtosis(y):.2f}")
```

| Observation | Action |
|:---|:---|
| Roughly symmetric (skewness between -0.5 and 0.5) | Use $y$ as-is |
| Right-skewed (skewness > 1) | Apply $\log(y)$ or $\sqrt{y}$. This often fixes heteroscedasticity too |
| Left-skewed (skewness < -1) | Apply $y^2$ or reflect then log |

| Observation (excess kurtosis) | Action |
|:---|:---|
| Between -1 and 1 (mesokurtic, normal-like tails) | No action needed |
| Between 1 and 3 (mildly leptokurtic, moderately heavy tails) | Check for outliers; usually manageable with RobustScaler or winsorizing at 1st/99th percentile |
| Greater than 3 (highly leptokurtic, very heavy tails) | Serious outlier concern; apply log/Box-Cox transform, use Huber loss or MAE instead of MSE, or remove extreme outliers after investigation |
| Less than -1 (platykurtic, light tails, fewer outliers) | Usually harmless for linear regression; no action required |

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
| Many outliers, or outliers carry meaningful signal | Keep them, use robust methods (RobustScaler, MAE as metric) |

### 1.4 Correlation and Scatter Plots

```python
import seaborn as sns
from scipy.stats import pearsonr

r, p_value = pearsonr(X, y)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X, y=y, edgecolor='k', alpha=0.7)
plt.title(f'YearsExperience vs Salary  (r = {r:.2f}, p = {p_value:.2e})')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()
plt.show()
```

Look for:
- **|r| > 0.7** -- strong linear relationship, good candidate for simple linear regression.
- **|r| between 0.3 and 0.7** -- moderate relationship; linear regression may work but expect lower R-squared.
- **|r| < 0.3** -- weak linear relationship; the feature may not be useful on its own, or the relationship is non-linear.
- **p-value < 0.05** -- the correlation is statistically significant, meaning you can reject the null hypothesis that r = 0 (the relationship is real, not due to chance). However, statistical significance alone does not imply a *strong* relationship. With large samples, even a small r can reach p < 0.05. Always pair the p-value with the magnitude of r.
- **Non-linear pattern** in the scatter plot -- flag for polynomial features in Step 3.

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
    X, y, test_size=0.2, random_state=42
)
```

**Critical rule:** Everything from this point forward (scaling, feature engineering, model fitting) is done on `X_train` / `y_train` only. The test set is touched only once, at the very end, for final evaluation. Fitting a scaler or imputer on the full dataset before splitting causes **data leakage**.

| Dataset size | Split ratio |
|:---|:---|
| < 1,000 samples | 80/20 and use cross-validation heavily |
| 1,000 - 100,000 | 80/20 is standard |
| > 100,000 | 90/10 or even 95/5 (test set is already large enough) |

---

## Step 3 -- Feature Engineering and Scaling

### 3.1 Multicollinearity Check (VIF)

> See [Multicollinearity](linear_regression_and_grad_descent_cheatsheet.md#multicollinearity) in the cheatsheet for full theory.

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
| VIF > 10 | **Must act.** Drop one feature from each correlated pair, apply PCA, or use Ridge regression (which handles multicollinearity natively) |

### 3.2 Feature Scaling

> See [Feature Scaling](../07_feature_engineering/feature_scaling.md) for the full guide on each method.

| Solving method | Scaling needed? | Why |
|:---|:---|:---|
| Normal Equation | No (optional) | Closed-form solution is scale-invariant |
| Gradient Descent | **Yes (mandatory)** | Unscaled features cause elongated contours and slow/failed convergence |

**Default choice: StandardScaler** (Z-score normalization). Use RobustScaler if you kept outliers.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use train's μ and σ
```

### 3.3 Polynomial Features (if needed)

Only add these if EDA (Step 1.4) revealed non-linear patterns in scatter plots.

> See [Polynomial Regression Cheatsheet](polynomial_regression_cheatsheet.md) for full theory and degree selection.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
```

| Situation | Degree |
|:---|:---|
| Mild curvature in one or two features | 2 |
| S-shaped or more complex curvature | 3 |
| > 3 | Almost always overfitting -- use regularization or switch models |

**After adding polynomial features, re-check VIF** -- polynomial terms are often correlated. If VIF explodes, pair with Ridge regularization.

---

## Step 4 -- Fit the Model

### 4.1 Choose the Solving Method

> See [Normal Equation vs Gradient Descent](linear_regression_and_grad_descent_cheatsheet.md#normal-equation-vs-gradient-descent) in the cheatsheet for full comparison.

| Scenario | Method |
|:---|:---|
| n_features < 10,000 and n_samples < 10,000,000 | Normal Equation (`LinearRegression()`) |
| Large dataset or many features | Gradient Descent (`SGDRegressor`) |
| $X^TX$ is singular (non-invertible) | Gradient Descent, or add regularization |

### 4.2 Fit the Baseline

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
```

Record training performance as a baseline:

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training R²:   {train_r2:.4f}")
```

---

## Step 5 -- Validate Assumptions

This is the most important diagnostic step. A model that violates assumptions may produce biased predictions or misleading statistical tests. Check each assumption, and follow the branching logic below.

> See [Key Assumptions](linear_regression_and_grad_descent_cheatsheet.md#key-assumptions) in the cheatsheet for the full theory behind each assumption.

### Assumption Validation Decision Tree

```
                          ┌─────────────┐
                          │  Fit Model   │
                          └──────┬───────┘
                                 ▼
                       ┌───────────────────┐
                       │  1. Linearity?    │
                       │  (residuals vs    │
                       │   fitted plot)    │
                       └───┬──────────┬────┘
                      Pass │          │ Fail: curved pattern
                           ▼          ▼
                     ┌──────────┐  Add poly terms or transform X
                     │          │  → go back to Step 3, refit
                     ▼          
               ┌───────────────────┐
               │  2. Independence? │
               │  (Durbin-Watson)  │
               └───┬──────────┬────┘
              Pass │          │ Fail: DW far from 2
                   ▼          ▼
             ┌──────────┐  Add lagged features or
             │          │  switch to time-series model
             ▼          
       ┌────────────────────────┐
       │  3. Homoscedasticity?  │
       │  (residuals vs fitted  │
       │   + Breusch-Pagan)     │
       └───┬───────────────┬────┘
      Pass │               │ Fail: funnel shape
           ▼               ▼
     ┌──────────┐  Transform y (log, sqrt),
     │          │  use WLS, or robust SE
     ▼          │  → refit
┌──────────────────┐
│  4. Normality?   │
│  (Q-Q plot +     │
│   Shapiro-Wilk)  │
└──┬─────────┬─────┘
   │         │
   ▼         ▼
  Pass    Fail ──► Is n > 30?
   │              ├── Yes: CLT applies, proceed (inference is approximate)
   ▼              └── No: Box-Cox transform y or remove outliers → refit
┌─────────────────────┐
│ Proceed to Step 6   │
│ (Evaluation)        │
└─────────────────────┘
```

### 5.1 Check Linearity

```python
import matplotlib.pyplot as plt

residuals = y_train - y_train_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs fitted values
axes[0].scatter(y_train_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Residuals vs each predictor (example for first feature)
axes[1].scatter(X_train_scaled[:, 0], residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Feature 0')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Feature 0')

plt.tight_layout()
plt.show()
```

| What you see | Diagnosis | Next step |
|:---|:---|:---|
| Random scatter around zero | Linearity holds | Move to 5.2 |
| U-shape or arch | Non-linear relationship | Add $x^2$ for the offending feature (Step 3.3), refit |
| S-shape | Cubic relationship | Add $x^3$, or apply log/sqrt transform to feature |
| Clear pattern only for one feature | That specific feature is non-linear | Transform or add polynomial terms for that feature only |

### 5.2 Check Independence

This is most relevant when data has a natural ordering (time series, sequential measurements).

```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print(f"Durbin-Watson: {dw:.2f}")
```

| DW value | Diagnosis | Next step |
|:---|:---|:---|
| 1.5 - 2.5 | No significant autocorrelation | Move to 5.3 |
| < 1.5 | Positive autocorrelation (common in time series) | Add lagged target/features, or switch to ARIMA/GLS |
| > 2.5 | Negative autocorrelation (rare) | Investigate data ordering, consider GLS |
| Data has no natural ordering (cross-sectional) | Independence is usually safe to assume | Move to 5.3 |

### 5.3 Check Homoscedasticity

Look at the same residuals-vs-fitted plot from 5.1, plus run a formal test.

```python
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

X_with_const = sm.add_constant(X_train_scaled)
bp_test = het_breuschpagan(residuals, X_with_const)
labels = ['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value']
print(dict(zip(labels, bp_test)))
```

| Observation | Diagnosis | Next step |
|:---|:---|:---|
| Random scatter, BP p-value > 0.05 | Homoscedasticity holds | Move to 5.4 |
| Funnel shape (spread increases with fitted values) | Heteroscedasticity | Apply $\log(y)$ → refit from Step 2 (re-split with transformed target) |
| Funnel shape but can't transform $y$ | Heteroscedasticity | Use Weighted Least Squares (WLS) or report robust standard errors (HC3) |
| BP p-value < 0.05 but plot looks OK | Borderline | Large samples can trigger significance easily; trust the plot more than the p-value |

### 5.4 Check Normality of Residuals

```python
from scipy import stats

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(residuals, bins=30, edgecolor='k', alpha=0.7, density=True)
x_range = np.linspace(residuals.min(), residuals.max(), 100)
axes[0].plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()), 'r-', lw=2)
axes[0].set_title('Residual Distribution')

# Q-Q plot
stats.probplot(residuals, plot=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# Formal test (best for n < 5000; for larger n, rely on Q-Q plot)
if len(residuals) < 5000:
    stat, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk: stat={stat:.4f}, p={p_value:.4f}")
```

| Observation | Diagnosis | Next step |
|:---|:---|:---|
| Points follow Q-Q diagonal, bell-shaped histogram | Normality holds | Proceed to Step 6 |
| Heavy tails (Q-Q curves away at ends) | Outliers inflating tails | Remove extreme outliers or apply Box-Cox to $y$ → refit |
| Right skew (Q-Q curves up at right end) | Skewed residuals | Apply $\log(y)$ → refit |
| Normality fails but $n > 30$ | CLT covers inference | Proceed to Step 6 (note: prediction intervals will be approximate) |
| Normality fails and $n \leq 30$ | Inference is unreliable | Transform $y$ (Box-Cox), remove outliers, or use bootstrapping for confidence intervals |

### Assumptions Summary: Quick Reference

| Assumption | Diagnostic tool | Pass criterion | What breaks if violated |
|:---|:---|:---|:---|
| Linearity | Residuals vs fitted plot | Random scatter | Biased predictions |
| Independence | Durbin-Watson test | DW ~ 2 | Inflated significance (false positives) |
| Homoscedasticity | Residuals vs fitted + Breusch-Pagan | No funnel, BP p > 0.05 | Wrong standard errors |
| Normality | Q-Q plot + Shapiro-Wilk | Points on diagonal, SW p > 0.05 | Invalid confidence intervals (less critical for large $n$) |

---

## Step 6 -- Evaluate Performance

> See [Regression Evaluation Metrics](../06_metrics/regression_metrics_cheatsheet.md) for detailed formulas, from-scratch code, and when to use each metric.

### 6.1 Test Set Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_test_pred = model.predict(X_test_scaled)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

n_features = X_test_scaled.shape[1]
n_samples = X_test_scaled.shape[0]
adj_r2 = 1 - (1 - test_r2) * (n_samples - 1) / (n_samples - n_features - 1)

print(f"Test RMSE:     {test_rmse:.4f}")
print(f"Test MAE:      {test_mae:.4f}")
print(f"Test R²:       {test_r2:.4f}")
print(f"Adjusted R²:   {adj_r2:.4f}")
```

### 6.2 Cross-Validation

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                            scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"CV RMSE: {cv_rmse.mean():.4f} +/- {cv_rmse.std():.4f}")
```

### 6.3 Diagnose the Results

Compare training metrics (Step 4.2) with test/CV metrics and use this decision table:

| Training R² | Test R² | Gap | CV std | Diagnosis | Action |
|:---|:---|:---|:---|:---|:---|
| < 0.5 | < 0.5 | Small | Low | **Underfitting** | Add features, add polynomial terms, reduce regularization |
| > 0.9 | < 0.7 | Large (> 0.15) | High | **Overfitting** | Add regularization (Step 7), reduce features, get more data |
| > 0.7 | > 0.7 | Small (< 0.05) | Low | **Good fit** | Proceed to Step 8 |
| > 0.7 | 0.5-0.7 | Moderate | Moderate | **Mild overfitting** | Try light regularization (small Ridge alpha), or remove weakest features |
| Any | Any | Any | > 0.1 * mean | **Unstable model** | Get more data, simplify the model, or increase $k$ in CV |

**Underfitting remedies** (go back to earlier steps):
- Add more relevant features → Step 1 / Step 3
- Add polynomial or interaction terms → Step 3.3
- Reduce regularization strength if already applied → Step 7
- Consider a non-linear model (trees, ensembles) → outside this guide

**Overfitting remedies** (proceed to Step 7):
- Apply Ridge, Lasso, or Elastic Net regularization → Step 7
- Remove irrelevant features (use Lasso for automatic selection)
- Collect more training data

---

## Step 7 -- Regularization (if needed)

Regularization is triggered by **overfitting** (Step 6) or **multicollinearity** (Step 3.1). It is not always needed -- skip this step if your baseline model from Step 4 already performs well.

> See [Regularization Techniques](../07_feature_engineering/regularization_techniques.md) for full theory on Ridge, Lasso, and Elastic Net.

### 7.1 Choose the Regularization Type

| Situation | Regularization | Why |
|:---|:---|:---|
| All features are believed relevant, multicollinearity present | **Ridge (L2)** | Shrinks all coefficients, never removes features, stabilizes correlated coefficients |
| Many features suspected to be irrelevant | **Lasso (L1)** | Drives irrelevant coefficients to exactly zero (automatic feature selection) |
| Correlated feature groups + some irrelevant features | **Elastic Net** | Combines Ridge's group stability with Lasso's sparsity |
| Not sure which to use | **Elastic Net** with `l1_ratio=0.5` | Safe default that blends both penalties |

### 7.2 Tune Hyperparameters via Cross-Validation

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Ridge
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge.fit(X_train_scaled, y_train)
print(f"Ridge best alpha: {ridge.alpha_}")

# Lasso
lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5)
lasso.fit(X_train_scaled, y_train)
print(f"Lasso best alpha: {lasso.alpha_}")
print(f"Lasso features selected: {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")

# Elastic Net
elastic = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5)
elastic.fit(X_train_scaled, y_train)
print(f"ElasticNet best alpha: {elastic.alpha_}, best l1_ratio: {elastic.l1_ratio_}")
```

### 7.3 Evaluate the Regularized Model

After fitting the regularized model, repeat Step 6 (test metrics + CV) and compare against the baseline from Step 4.

```python
for name, m in [("Baseline", model), ("Ridge", ridge), ("Lasso", lasso), ("ElasticNet", elastic)]:
    y_pred = m.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name:12s} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
```

| Outcome | Interpretation | Next step |
|:---|:---|:---|
| Regularized R² > Baseline R² on test set | Regularization helped reduce overfitting | Keep the regularized model |
| Regularized R² ~ Baseline R² | Model was not overfitting much | Use whichever is simpler; Ridge if multicollinearity was an issue |
| Regularized R² < Baseline R² | Regularization is too strong or wrong type | Try a smaller alpha, or a different regularization type |
| Lasso set most coefficients to zero | Most features are irrelevant | Rebuild model with only the selected features for interpretability |

After choosing the best regularized model, **re-validate assumptions** (Step 5) on the new residuals to confirm the remedies hold.

---

## Step 8 -- Final Model and Reporting

### 8.1 Retrain on Full Training Set

The best model (with its tuned hyperparameters) is already trained on the training set. If you used cross-validation only for hyperparameter selection, the final `fit()` call from Step 7.2 already used the full training set.

### 8.2 Final Test Evaluation

```python
best_model = ridge  # or whichever performed best

y_final_pred = best_model.predict(X_test_scaled)
final_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
final_mae = mean_absolute_error(y_test, y_final_pred)
final_r2 = r2_score(y_test, y_final_pred)

print("=== Final Model Performance ===")
print(f"RMSE:         {final_rmse:.4f}")
print(f"MAE:          {final_mae:.4f}")
print(f"R²:           {final_r2:.4f}")
```

### 8.3 Inspect Coefficients

```python
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'x{i}' for i in range(X_train.shape[1])]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': best_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df.to_string(index=False))
print(f"\nIntercept: {best_model.intercept_:.4f}")
```

### 8.4 Reporting Checklist

Document these items for reproducibility and transparency:

| Item | What to record |
|:---|:---|
| **Data** | Source, size ($m$ samples, $n$ features), date range, any filters applied |
| **Preprocessing** | Missing value strategy, outlier handling, scaling method, feature transformations |
| **Assumption checks** | Which passed, which failed, what remedies were applied |
| **Model** | Type (OLS / Ridge / Lasso / ElasticNet), hyperparameters ($\alpha$, `l1_ratio`) |
| **Performance** | RMSE, MAE, R², Adjusted R² on the held-out test set |
| **Cross-validation** | CV RMSE mean and standard deviation |
| **Limitations** | Any violated assumptions that could not be fully remedied (e.g., "normality failed; confidence intervals are approximate") |
| **Feature importance** | Top coefficients (positive and negative) and their magnitudes |

---

## Full Workflow Summary

```
Step 0: Define the Problem
  └─ Is target continuous? → Yes → proceed. No → use classification.

Step 1: EDA
  ├─ Target skewed?      → transform y
  ├─ Missing values?     → impute or drop
  ├─ Outliers?           → remove, cap, or keep
  ├─ Correlations > 0.8? → flag for VIF check
  └─ Non-linear scatter? → flag for polynomial features

Step 2: Train/Test Split
  └─ All fitting from here uses training set only

Step 3: Feature Engineering
  ├─ VIF check           → drop, combine, PCA, or use Ridge
  ├─ Scale features      → StandardScaler (mandatory for gradient descent)
  └─ Polynomial features → only if non-linearity flagged in Step 1

Step 4: Fit Baseline Model
  └─ LinearRegression() → record training RMSE and R²

Step 5: Validate Assumptions
  ├─ Linearity fails?         → add poly terms → refit (Step 3)
  ├─ Independence fails?      → add lags or switch to time-series
  ├─ Homoscedasticity fails?  → transform y or use WLS → refit
  └─ Normality fails?         → if n>30, proceed; else Box-Cox → refit

Step 6: Evaluate Performance
  ├─ Underfitting?  → add features/complexity (Step 3)
  ├─ Overfitting?   → regularize (Step 7)
  └─ Good fit?      → proceed to Step 8

Step 7: Regularization (if needed)
  ├─ All features relevant + multicollinearity → Ridge
  ├─ Many irrelevant features                 → Lasso
  └─ Correlated groups + irrelevant features  → Elastic Net

Step 8: Final Evaluation and Reporting
  └─ Report metrics, coefficients, limitations, assumptions status
```
