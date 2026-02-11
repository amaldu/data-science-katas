# Regression Evaluation Metrics

Evaluation metrics quantify how well the model's predictions match the actual values. Different metrics emphasize different aspects of error, so choosing the right one depends on the problem.

Where: $m$ = number of samples, $y_i$ = actual value, $\hat{y}_i$ = predicted value, $\bar{y}$ = mean of actual values.

---

## 1. Mean Squared Error (MSE)

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

**Time Complexity:** $O(m)$ — single pass over predictions.

**Python:**
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)

# From scratch
mse = np.mean((y_true - y_pred) ** 2)
```

---

## 2. Root Mean Squared Error (RMSE)

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

**Time Complexity:** $O(m)$ — single pass plus one square root.

**Python:**
```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_true, y_pred)

# From scratch
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
```

---

## 3. Mean Absolute Error (MAE)

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

**Time Complexity:** $O(m)$ — single pass over predictions.

**MSE vs MAE Example:**
```
Predictions errors: [1, 1, 1, 10]

MAE  = (1 + 1 + 1 + 10) / 4 = 3.25
RMSE = √((1 + 1 + 1 + 100) / 4) = √25.75 ≈ 5.07

→ RMSE is much larger because the outlier (10) gets squared to 100.
  Choose MAE if that outlier shouldn't dominate the evaluation.
  Choose RMSE if that large error is genuinely important to penalize.
```

**Python:**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)

# From scratch
mae = np.mean(np.abs(y_true - y_pred))
```

---

## 4. R² Score (Coefficient of Determination)

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

![R² Values Illustration](images/08_r2_values.png)

*R² shows the proportion of variance explained. Higher R² = data points cluster tighter around the regression line.*

![Coefficient of Determination](images/09_coefficient_of_determination.png)

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

**Time Complexity:** $O(m)$ — two passes (compute mean, then sums).

**Python:**
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)

# From scratch
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - (ss_res / ss_tot)
```

---

## 5. Adjusted R²

**Definition:** A modified version of R² that penalizes the addition of irrelevant features. Unlike R², it can decrease if a new feature doesn't improve the model enough to justify its added complexity.

$$R^2_{adj} = 1 - \frac{(1 - R^2)(m - 1)}{m - n - 1}$$

Where $m$ = samples, $n$ = features.

**Interpretation:**
- Same scale as R² (0 to 1 for reasonable models)
- If adding a feature increases Adjusted R² → the feature is useful
- If adding a feature decreases Adjusted R² → the feature adds noise, not value

**When to use:**
- **Always** when comparing models with different numbers of features
- Multiple regression with many candidate features
- Feature selection: only keep features that improve Adjusted R²

**Advantages:**
- ✅ Accounts for the number of features (penalizes overfitting)
- ✅ Better for model comparison than raw R²
- ✅ Decreases when irrelevant features are added

**Disadvantages:**
- ❌ Less intuitive than R²
- ❌ Can be negative for very poor models
- ❌ Assumes the correct model form (linear)

**Python:**
```python
# From scratch (sklearn does not have a built-in adjusted R²)
def adjusted_r2(r2, m, n):
    """m = samples, n = features"""
    return 1 - (1 - r2) * (m - 1) / (m - n - 1)

r2 = r2_score(y_true, y_pred)
adj_r2 = adjusted_r2(r2, m=len(y_true), n=X.shape[1])
```

---

## 6. Mean Absolute Percentage Error (MAPE)

**Definition:** Measures the average absolute error as a **percentage** of the actual values. Useful for comparing accuracy across datasets with different scales.

$$\text{MAPE} = \frac{100}{m} \sum_{i=1}^{m} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Interpretation:**
- Units: percentage (%)
- "On average, our predictions are off by MAPE%"
- Example: MAPE = 8% means predictions deviate by 8% on average

**When to use:**
- When you need a **scale-independent** error measure
- When stakeholders want errors expressed as percentages
- Comparing models across different datasets or target scales

**Advantages:**
- ✅ Scale-independent → can compare across different datasets
- ✅ Easy to communicate to non-technical stakeholders ("8% error")
- ✅ Intuitive percentage interpretation

**Disadvantages:**
- ❌ **Undefined when $y_i = 0$** (division by zero)
- ❌ **Asymmetric:** penalizes over-predictions more than under-predictions for the same absolute error
- ❌ Can be misleadingly large when actual values are very small
- ❌ Not suitable for data that contains or crosses zero

**Python:**
```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred)

# From scratch
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

---

## 7. Median Absolute Error (MedAE)

**Definition:** The median of all absolute differences between predicted and actual values. Even more robust to outliers than MAE.

$$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \ldots, |y_m - \hat{y}_m|)$$

**Interpretation:**
- Units: same as target variable
- "Half of our predictions have an error less than MedAE"
- Completely ignores the magnitude of the worst predictions

**When to use:**
- When your data has **significant outliers** that you want to ignore in evaluation
- When you want the most robust measure of typical prediction error

**Advantages:**
- ✅ Extremely robust to outliers (unaffected by any single extreme error)
- ✅ Same units as target → interpretable
- ✅ Represents typical error better than MAE/RMSE in skewed data

**Disadvantages:**
- ❌ Ignores the tails — doesn't tell you about worst-case performance
- ❌ Not differentiable (cannot be used as a loss function)
- ❌ Less commonly used and reported

**Python:**
```python
from sklearn.metrics import median_absolute_error

medae = median_absolute_error(y_true, y_pred)

# From scratch
medae = np.median(np.abs(y_true - y_pred))
```

---

## How to Choose the Right Metric

| Question | Recommended Metric |
|:---|:---|
| Need a metric for **optimization/training**? | **MSE** (differentiable, convex) |
| Need to **report** error to stakeholders? | **RMSE** (interpretable units) or **MAPE** (percentage) |
| Data has **outliers**? | **MAE** or **MedAE** |
| Comparing models with **different feature counts**? | **Adjusted R²** |
| Comparing models on the **same dataset**? | **R²** or **RMSE** |
| Comparing across **different scales**? | **MAPE** or **R²** |
| Need **worst-case** awareness? | **RMSE** (penalizes large errors) |
| Large errors are **especially costly**? | **MSE** or **RMSE** |
| All errors are **equally costly**? | **MAE** |

**Best practice:** Never rely on a single metric. Report at least 2–3 metrics to get a complete picture:
- **R²** (overall explanatory power) + **RMSE** (error magnitude) + **MAE** (robust error)

---

## Evaluation Metrics Summary

| Metric | Formula | Units | Outlier Sensitive? | Interpretability | Best For |
|:---|:---|:---|:---|:---|:---|
| **MSE** | $\frac{1}{m}\sum(y_i - \hat{y}_i)^2$ | Squared | ❌ Very sensitive | Low | Optimization / training |
| **RMSE** | $\sqrt{\text{MSE}}$ | Original | ❌ Sensitive | High | Reporting, comparing models |
| **MAE** | $\frac{1}{m}\sum|y_i - \hat{y}_i|$ | Original | ✅ Robust | High | Robust evaluation |
| **R²** | $1 - SS_{res}/SS_{tot}$ | Unitless | ❌ Sensitive | High | Explanatory power |
| **Adjusted R²** | Penalized R² | Unitless | ❌ Sensitive | High | Model comparison (different # features) |
| **MAPE** | $\frac{100}{m}\sum|{(y_i - \hat{y}_i)}/{y_i}|$ | Percentage | ❌ Sensitive | Very high | Cross-scale comparison |
| **MedAE** | $\text{median}(|y_i - \hat{y}_i|)$ | Original | ✅ Very robust | High | Outlier-heavy data |

---

## Python Quick Reference

```python
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)

# All metrics in one go
print(f"MSE:    {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE:   {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE:    {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R²:     {r2_score(y_true, y_pred):.4f}")
print(f"MAPE:   {mean_absolute_percentage_error(y_true, y_pred):.4f}")
print(f"MedAE:  {median_absolute_error(y_true, y_pred):.4f}")
```
