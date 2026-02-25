# Polynomial Regression Cheatsheet

> Polynomial Regression extends Linear Regression to model non-linear relationships by adding polynomial terms to the feature set. It remains a linear model in the parameters.

---

## What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression that models **non-linear relationships** between features and the target variable by adding polynomial terms (powers and interactions of the original features) to the model.

Despite fitting curves, Polynomial Regression is still a **linear model**. It is linear in the **parameters** $\theta$, not in the features $x$. We simply create new features ($x^2, x^3, \ldots$) and then fit a standard linear model on the expanded feature set.

---

## Equation

For a single feature $x$, polynomial regression of degree $d$:

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \cdots + \theta_d x^d$$

In matrix form, we transform the feature matrix $X$ into a polynomial feature matrix $X_{poly}$ and then apply regular linear regression:

$$\hat{y} = X_{poly} \theta$$

---

## Visual Intuition

![Polynomial Regression Example](../../images/06_polynomial_regression.png)

*Polynomial regression fits a curve to data that a straight line cannot capture. Higher degrees fit more complex patterns but risk overfitting.*

---

## How It Works

```
Step 1: Choose a polynomial degree d
Step 2: Transform features: [x] → [x, x², x³, ..., xᵈ]
Step 3: (Optional) Add interaction terms for multiple features
Step 4: Fit standard Linear Regression on the transformed features
Step 5: Use Normal Equation or Gradient Descent to find θ
```

**For multiple features** ($x_1, x_2$) with degree 2, the expanded features include:

$$[1, \; x_1, \; x_2, \; x_1^2, \; x_1 x_2, \; x_2^2]$$

The number of features grows combinatorially: with $n$ original features and degree $d$, the expanded feature count is $\binom{n+d}{d}$. For example:

| Original features ($n$) | Degree ($d$) | Polynomial features |
|:---:|:---:|:---:|
| 2 | 2 | 6 |
| 2 | 3 | 10 |
| 10 | 2 | 66 |
| 10 | 3 | 286 |
| 100 | 2 | 5,151 |

---

## Choosing the Polynomial Degree

The degree $d$ is a **hyperparameter** that controls the model's complexity:

| Degree | Model behavior | Risk |
|:---:|:---|:---|
| **1** | Standard linear regression (straight line) | Underfitting if relationship is non-linear |
| **2–3** | Captures moderate curvature | Usually a good default |
| **4–6** | Fits complex shapes | Starting to overfit on small datasets |
| **> 6** | Extreme flexibility | Almost certainly overfitting |

**How to choose:**
- Use **cross-validation** to compare different degrees and pick the one with the lowest validation error.
- Watch the **learning curves**: if training error is low but validation error is high, the degree is too high.
- **Start simple** (degree 2) and increase only if the model underfits.

---

## Overfitting and Regularization

Polynomial Regression is highly prone to overfitting because:
1. **High-degree polynomials** can fit arbitrarily complex curves, including noise.
2. **Coefficient magnitudes** can become very large, leading to wild oscillations between data points.

![Underfitting vs Overfitting](../../images/07_overfitting.png)

*Left: Underfitting (degree too low). Right: Overfitting (degree too high). The model fits training data perfectly but generalizes poorly.*

**Solutions:**
- **Regularization** (Ridge, Lasso, Elastic Net). Constrains coefficient magnitudes. See [regularization_techniques.md](../regularization_techniques.md).
- **Reduce degree**. Simplest fix.
- **More training data**. Harder for the model to memorize.
- **Feature scaling**. Essential when using polynomial features because $x^d$ magnifies scale differences enormously.

---

## When to Use Polynomial Regression Instead of Other Linear Models

Polynomial Regression sits between plain Linear Regression and fully non-linear models. Choosing it over other options depends on the data, the problem, and what you need from the model.

**Use Polynomial Regression when:**

1. **The relationship is clearly non-linear but smooth.** If a scatter plot of feature vs. target shows a curve (parabola, S-shape, or wave), a straight line will systematically underfit. Polynomial terms can capture that curvature without leaving the linear regression framework.

2. **You have few features.** Polynomial feature expansion grows combinatorially. With 2–5 original features and degree 2–3, it's manageable. With 100 features, even degree 2 produces 5,000+ features. At that point, other models are better choices.

3. **You need interpretable coefficients.** The model is still $\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \ldots$. Each coefficient has a clear meaning (the effect of that term on the prediction). Tree-based models and neural networks don't offer this.

4. **You need to extrapolate (with caution).** Unlike tree-based models which predict a constant outside the training range, polynomial models produce a mathematical function that extends beyond the data. This can be useful in engineering or physics where you know the underlying relationship is polynomial. But dangerous otherwise, because high-degree polynomials diverge wildly.

---

## Common Pitfalls

1. **Using too high a polynomial degree.** It is tempting to keep increasing the degree to minimize training error, but this almost always leads to overfitting. The model will fit noise rather than the true pattern, and predictions outside the training range become wildly inaccurate.
2. **Forgetting to scale features.** Polynomial features amplify scale differences enormously. Without scaling, gradient descent may fail to converge, numerical instability can cause overflow, and regularization will penalize features unfairly based on their scale rather than their importance.
3. **Ignoring multicollinearity in polynomial terms.** Polynomial features are inherently correlated with each other ($x$ and $x^2$ share information). This inflates VIF and makes coefficients unstable. Always pair polynomial features with regularization (Ridge or Elastic Net).
4. **Not using regularization.** Plain polynomial regression without regularization is almost guaranteed to overfit when the degree exceeds 2-3. Always combine polynomial features with Ridge, Lasso, or Elastic Net.
5. **Trusting extrapolation.** Polynomial models can produce wildly wrong predictions outside the training data range. A degree-5 polynomial that fits beautifully within $[0, 10]$ can output absurd values at $x = 11$. Never extrapolate with high-degree polynomials unless the underlying relationship is known from domain knowledge.

## Pros and Cons

✅ Captures non-linear relationships while remaining a linear model in the parameters
✅ Interpretable coefficients (each term has a clear meaning)
✅ Easy to implement using standard linear regression tools
✅ Can extrapolate (with caution) beyond the training range
✅ Works well with few features and low polynomial degrees

❌ Feature count grows combinatorially with degree and number of original features
❌ Highly prone to overfitting at higher degrees
❌ Extrapolation is dangerous (polynomials diverge wildly outside training range)
❌ Requires feature scaling (polynomial terms amplify scale differences)
❌ Not suitable for datasets with many features (feature explosion)

## Key Takeaways

| Aspect | Detail |
|:---|:---|
| **What it does** | Adds polynomial terms to model non-linear relationships |
| **Still linear?** | Yes. Linear in parameters, not in features |
| **Main hyperparameter** | Polynomial degree $d$ |
| **Danger** | Overfitting increases sharply with higher degrees |
| **Must-do** | Feature scaling (polynomial terms amplify scale differences) |
| **Best practice** | Use with regularization (Ridge or Elastic Net) |
| **Feature explosion** | $n=100, d=2$ → 5,151 features. Be careful with high $n$ or $d$ |
