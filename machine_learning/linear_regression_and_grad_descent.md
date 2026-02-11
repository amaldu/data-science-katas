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

#### Matrix Form

$$\hat{y} = X\theta$$

Where:
- $X$ = feature matrix $(m \times n)$, m samples, n features
- $\theta$ = parameter vector $[\beta_0, \beta_1, \ldots, \beta_n]^T$
- $\hat{y}$ = predicted values

### Visual Intuition

![Linear Regression Best Fit Line](images/01_linear_regression_best_fit.png)

*The goal is to find the line that minimizes the total distance between data points and the line.*

### Estimation Method

Linear Regression uses **Ordinary Least Squares (OLS)** as its estimation method. OLS finds the parameter values that **minimize the sum of squared residuals** — the squared vertical distances between each observed data point and the predicted value on the regression line.

**Why "Least Squares"?** The method minimizes the sum of squared errors rather than absolute errors. Squaring has two advantages: it makes all errors positive (so they don't cancel out), and it penalizes large errors more heavily, encouraging the model to avoid big mistakes.

**How it works:** Given the data, OLS computes the parameters $\theta$ (intercept and coefficients) such that the total squared error between the predicted values $\hat{y}_i = X_i \theta$ and the actual values $y_i$ is as small as possible. This can be solved either analytically (Normal Equation) or iteratively (Gradient Descent) — both approaches are detailed below.



## Cost Function

**Mean Squared Error (MSE):**

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

MSE is the standard metric that measures the average of the squared differences between predicted and actual values.

**Optimization Cost Function $J(\theta)$:**

For gradient descent, we use a slightly modified version with a $\frac{1}{2}$ factor:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

Or in matrix form:

$$J(\theta) = \frac{1}{2m} (y - X\theta)^T (y - X\theta)$$

The $\frac{1}{2}$ is a mathematical convenience — when we take the derivative of $J(\theta)$ during gradient descent, the exponent 2 from the square cancels with the $\frac{1}{2}$, producing a cleaner gradient formula. **It does not change the location of the minimum** (multiplying a function by a constant doesn't move its minimum), so the optimal $\theta$ is the same whether we minimize MSE or $J(\theta)$.

![Convex Cost Function](images/02_convex_cost_function.png)

*The MSE cost function for linear regression is CONVEX (bowl-shaped). It has exactly ONE global minimum — no local minima to worry about.*

## Key Assumptions

Linear regression relies on four critical assumptions. If these assumptions are violated, the model's parameter estimates, predictions, or statistical tests (p-values, confidence intervals) may be unreliable.

---

### 1. Linearity

**What it means:** The relationship between the independent variables $X$ and the dependent variable $y$ is linear — the expected value of $y$ changes at a constant rate as each feature $x_j$ changes, holding all other features constant.

**Why it matters:** The entire linear regression model is built on the equation $y = X\theta + \varepsilon$. If the true relationship is curved, quadratic, or otherwise non-linear, a straight line will systematically miss the pattern. The model will underfit and the residuals will show clear structure instead of being random.

**How to detect violations:**
- Plot residuals vs. predicted values — if you see a curve or U-shape instead of a random scatter, the linearity assumption is violated.
- Plot each feature against the target to visually inspect the relationship.

**What to do if violated:**
- Add polynomial features ($x^2, x^3$) — see the Polynomial Regression section.
- Apply non-linear transformations to features (e.g., $\log(x)$, $\sqrt{x}$).
- Use a non-linear model instead (decision trees, neural networks).

---

### 2. Independence

**What it means:** The observations (data points) are independent of each other — the value of one observation does not influence or predict the value of another. In particular, the residual (error) of one observation should not be correlated with the residual of another.

**Why it matters:** If observations are correlated (e.g., time series data where today's value depends on yesterday's), the model underestimates the true variance of the coefficients. This makes standard errors too small, p-values too low, and confidence intervals too narrow — you think your results are more significant than they actually are.

**How to detect violations:**
- **Durbin-Watson test** — tests for autocorrelation in residuals. Values near 2 indicate no autocorrelation; values near 0 or 4 indicate positive or negative autocorrelation.
- Plot residuals in order — if you see patterns (waves, trends), the observations are not independent.

**What to do if violated:**
- Use time series models (ARIMA, exponential smoothing) instead.
- Add lagged variables as features.
- Use Generalized Least Squares (GLS) which accounts for correlated errors.

---

### 3. Homoscedasticity (Constant Variance)

**What it means:** The variance of the residuals (errors) is **constant** across all levels of the predicted values. In other words, the spread of the errors doesn't change — the model is equally uncertain about its predictions whether the target value is small or large.

The opposite — **heteroscedasticity** — means the error variance changes. A common example: predicting income, where prediction errors are small for low incomes but very large for high incomes.

**Why it matters:** OLS assumes equal variance to give equal weight to all observations. If variance is larger for some observations, those high-variance points disproportionately influence the model. The coefficient estimates remain unbiased, but:
- Standard errors are wrong → hypothesis tests and confidence intervals are unreliable.
- The model is not efficient (not the best possible estimator).

**How to detect violations:**
- **Residuals vs. fitted values plot** — if the residuals fan out (funnel shape) or form a pattern, variance is not constant.
- **Breusch-Pagan test** or **White's test** — formal statistical tests for heteroscedasticity.

**What to do if violated:**
- Apply a transformation to the target variable (e.g., $\log(y)$, $\sqrt{y}$) to stabilize variance.
- Use **Weighted Least Squares (WLS)** — gives less weight to observations with higher variance.
- Use **robust standard errors** (heteroscedasticity-consistent standard errors) which give correct p-values even with non-constant variance.

---

### 4. Normality of Residuals

**What it means:** The residuals $\varepsilon_i = y_i - \hat{y}_i$ follow a **normal (Gaussian) distribution** with mean zero and constant variance: $\varepsilon \sim \mathcal{N}(0, \sigma^2)$.

**Why it matters:** The normality assumption is needed for **statistical inference** — specifically for p-values, t-tests on coefficients, and confidence intervals to be valid. It does **not** affect the OLS estimates themselves (OLS finds the best linear fit regardless of residual distribution), but without normality:
- You cannot trust whether a coefficient is statistically significant.
- Confidence intervals may be too wide or too narrow.
- Prediction intervals will be inaccurate.

**Important nuance:** For large samples ($m > 30$), the **Central Limit Theorem** makes the coefficient estimates approximately normal even if the residuals are not. So normality is most critical for small sample sizes.

**How to detect violations:**
- **Q-Q plot** (quantile-quantile plot) — residuals should fall approximately on a straight line.
- **Shapiro-Wilk test** or **Kolmogorov-Smirnov test** — formal tests for normality.
- **Histogram of residuals** — should look roughly bell-shaped.

**What to do if violated:**
- Transform the target variable ($\log(y)$, Box-Cox transformation).
- Remove outliers that distort the distribution.
- For large samples, this assumption is less critical due to the Central Limit Theorem.
- Use non-parametric methods or bootstrapping for inference.

---

### Assumptions Summary

| Assumption | What It Affects If Violated | Severity |
|:---|:---|:---|
| **Linearity** | Biased predictions, systematic errors | High — model is fundamentally wrong |
| **Independence** | Standard errors, p-values, confidence intervals | High — conclusions are invalid |
| **Homoscedasticity** | Standard errors, efficiency of estimates | Medium — estimates are OK, but inference is off |
| **Normality** | p-values, confidence intervals, prediction intervals | Low for large samples (CLT helps) |

---

## Two Main Approaches to Find Optimal Parameters

### 1. Normal Equation (Closed-Form Solution)

The Normal Equation is a **direct analytical formula** that computes the optimal parameters $\theta$ in one step by setting the derivative of the cost function $J(\theta)$ to zero and solving for $\theta$ algebraically.

**Time Complexity:** $O(n^3 + mn^2)$ — computing $X^TX$ costs $O(mn^2)$ and inverting it costs $O(n^3)$. **Space:** $O(mn + n^2)$.

**Derivation intuition:**

We want to find $\theta$ that minimizes $J(\theta) = \frac{1}{2m}(X\theta - y)^T(X\theta - y)$. Taking the gradient and setting it to zero:

$$\nabla_\theta J(\theta) = \frac{1}{m} X^T(X\theta - y) = 0$$

Solving for $\theta$:

$$X^T X\theta = X^T y$$

$$\theta = (X^T X)^{-1} X^T y$$

This is called the "Normal Equation" because it comes from setting the gradient **normal** (perpendicular) to the error — the residual vector $(X\theta - y)$ is orthogonal to the column space of $X$ at the optimal solution.

**Formula:**

$$\theta = (X^T X)^{-1} X^T y$$

Where:
- $X$ = feature matrix $(m \times n)$ with intercept column of ones
- $X^T$ = transpose of $X$
- $(X^T X)^{-1}$ = inverse of the $n \times n$ matrix $X^T X$
- $y$ = target vector $(m \times 1)$
- $\theta$ = resulting parameter vector $(n \times 1)$

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

![Gradient Descent Convergence](images/03_gradient_descent_convergence.gif)

*Gradient descent with different initial conditions, iteratively stepping toward the minimum. Each step moves $\theta$ in the direction of steepest descent, with step size controlled by the learning rate ($\alpha$) and the gradient magnitude.*

#### Update Rule

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

#### Gradient

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \, x_{ij}$$$$

#### Vectorized Update

$$\theta := \theta - \frac{\alpha}{m} X^T (X\theta - y)$$

#### The Learning Rate ($\alpha$)

The learning rate $\alpha$ is a **hyperparameter** — a value that is set before training begins and is not learned from the data. It controls the size of each step during gradient descent. Choosing it correctly is critical.

![Learning Rate Effects](images/04_learning_rate_effects.png)

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

---

##### 1. Batch Gradient Descent (BGD)

**Definition:** Batch Gradient Descent computes the gradient of the cost function using the **entire training dataset** at every single update step. It sums the errors across all $m$ samples, calculates the average gradient, and then updates the parameters once per epoch.

**Time Complexity:** $O(mn)$ per iteration, $O(kmn)$ total — where $k$ = number of iterations. **Space:** $O(mn)$ (full dataset in memory).

**Update rule:**

$$\theta := \theta - \frac{\alpha}{m} \sum_{i=1}^{m} \nabla_\theta L(\hat{y}_i, y_i)$$

**How it works:**
1. Use all $m$ training samples to compute predictions.
2. Calculate the error for every sample.
3. Average all the gradients into one gradient vector.
4. Update $\theta$ once.
5. Repeat until convergence.

**Advantages:**
- ✅ **Stable convergence** — the gradient is exact (no noise), so the cost decreases smoothly at every step.
- ✅ **Guaranteed to converge** to the global minimum for convex problems (with appropriate learning rate).
- ✅ **Deterministic** — same result every run given the same initialization.
- ✅ **Efficient vectorization** — a single matrix multiplication computes the gradient for all samples at once.

**Disadvantages:**
- ❌ **Slow for large datasets** — each update requires processing all $m$ samples. If $m = 10$ million, every single step is expensive.
- ❌ **High memory usage** — the entire dataset must fit in memory to compute the gradient.
- ❌ **Cannot learn online** — cannot incorporate new data without retraining on the entire dataset.
- ❌ **Can get stuck in saddle points** (in non-convex problems) because the gradient is too smooth to escape.

**When to use:** Small to medium datasets (up to ~10,000 samples) where stability is more important than speed.

---

##### 2. Stochastic Gradient Descent (SGD)

**Definition:** Stochastic Gradient Descent updates the parameters after computing the gradient on **a single randomly chosen training sample**. Instead of waiting to see all the data, it makes a small update after each example.

**Time Complexity:** $O(n)$ per update, $O(mn)$ per epoch, $O(kmn)$ total. **Space:** $O(n)$ (one sample at a time).

**Update rule (for one random sample $i$):**

$$\theta := \theta - \alpha \, \nabla_\theta L(\hat{y}_i, y_i)$$

**How it works:**
1. Shuffle the training data randomly.
2. Pick one sample $(x_i, y_i)$.
3. Compute the gradient using only that sample.
4. Update $\theta$ immediately.
5. Move to the next sample. One pass through all samples = one epoch.

**Advantages:**
- ✅ **Very fast updates** — each step is $O(n)$ instead of $O(mn)$, making it much faster per iteration.
- ✅ **Low memory** — only one sample is needed in memory at a time.
- ✅ **Supports online learning** — can learn from new data as it arrives without retraining from scratch.
- ✅ **Can escape shallow local minima and saddle points** — the noise from random sampling helps the algorithm explore and avoid getting stuck (beneficial for non-convex problems like neural networks).

**Disadvantages:**
- ❌ **Noisy, unstable convergence** — the gradient from one sample is a very noisy estimate of the true gradient, causing the parameters to zigzag erratically.
- ❌ **Never truly converges** — oscillates around the minimum instead of settling on it (unless the learning rate is decayed).
- ❌ **Cannot exploit vectorized hardware** — processing one sample at a time doesn't benefit from GPU/matrix parallelism.
- ❌ **Sensitive to learning rate** — too high causes divergence, too low negates the speed advantage.

**When to use:** Very large datasets, online learning scenarios, or when you need to escape local minima in non-convex optimization.

---

##### 3. Mini-Batch Gradient Descent

**Definition:** Mini-Batch Gradient Descent is the **compromise between Batch GD and SGD**. It computes the gradient on a small random subset (mini-batch) of the training data — typically 32 to 256 samples — and updates the parameters once per mini-batch.

**Time Complexity:** $O(bn)$ per update, $O(mn)$ per epoch, $O(kmn)$ total — where $b$ = batch size. **Space:** $O(bn)$.

**Update rule (for a mini-batch $B$ of size $b$):**

$$\theta := \theta - \frac{\alpha}{b} \sum_{i \in B} \nabla_\theta L(\hat{y}_i, y_i)$$

**How it works:**
1. Shuffle the training data.
2. Split it into mini-batches of size $b$ (e.g., 32, 64, 128, 256).
3. For each mini-batch: compute gradient, update $\theta$.
4. One pass through all mini-batches = one epoch.

**Common mini-batch sizes:**

| Batch size | Trade-off |
|:---:|:---|
| **32** | More noise, more updates per epoch, better generalization |
| **64–128** | Good balance (most common default) |
| **256** | Smoother gradient, fewer updates, faster per epoch on GPU |
| **512+** | Very smooth, but may generalize worse and needs more memory |

**How to choose the mini-batch size:**

The batch size is a hyperparameter that affects training speed, convergence behavior, and generalization. Here is how to choose it:

**1. Start with 32 or 64.** These are the most widely used defaults. Research (Bengio, 2012; Masters & Luschi, 2018) has consistently shown that smaller batch sizes (32–64) generalize better than large ones. When in doubt, use 32.

**2. Use powers of 2.** Batch sizes of 32, 64, 128, 256, 512 align with GPU memory architecture. Non-power-of-2 sizes waste hardware capacity and run slower.

**3. Consider your dataset size.** The batch size should be smaller than the training set, but the ratio matters:

| Dataset size ($m$) | Recommended batch size |
|:---:|:---|
| $m < 500$ | Use full batch (Batch GD) — dataset is small enough |
| $500 < m < 5{,}000$ | 32–64 |
| $5{,}000 < m < 100{,}000$ | 64–256 |
| $m > 100{,}000$ | 128–512 (limited by GPU memory) |

**4. Check your GPU memory.** The practical upper limit on batch size is how much data fits in your GPU (or RAM). If you run out of memory, reduce the batch size. The formula is roughly:

$$b_{\max} \approx \frac{\text{Available GPU memory}}{\text{Memory per sample} \times \text{multiplier for gradients/activations}}$$

**5. Understand the trade-off between batch size and learning rate:**

| Batch size | Gradient noise | Learning rate |
|:---:|:---|:---|
| Small (32) | High noise → acts as regularization | Use a smaller learning rate |
| Large (512) | Low noise → more precise gradient | Can use a larger learning rate |

A common rule of thumb: **when you double the batch size, multiply the learning rate by $\sqrt{2}$** (linear scaling rule). This keeps the effective noise level roughly constant.

**6. Larger batches ≠ better models.** A counter-intuitive finding in deep learning research: larger batch sizes often lead to **worse** generalization. The noise from small batches helps the optimizer escape sharp, narrow minima and find flatter minima that generalize better. This is called the **generalization gap**.

**Quick decision:**
- **Default choice:** 32 or 64 — works well in almost all cases.
- **If training is too slow:** increase to 128 or 256 (and increase learning rate proportionally).
- **If running out of memory:** decrease batch size.
- **If model overfits:** try a smaller batch size (more noise = implicit regularization).
- **If loss is very noisy / unstable:** increase batch size for smoother gradients.

**Advantages:**
- ✅ **Balances speed and stability** — smoother than SGD, faster than Batch GD.
- ✅ **Exploits vectorized hardware** — GPUs and modern CPUs are optimized for matrix operations on batches, making mini-batch much faster than processing one sample at a time.
- ✅ **Good generalization** — the moderate noise from mini-batches acts as a form of regularization, often leading to better models than Batch GD.
- ✅ **Scalable** — works well for datasets of any size.

**Disadvantages:**
- ❌ **Extra hyperparameter** — you need to choose the batch size $b$ in addition to the learning rate.
- ❌ **Still oscillates** (less than SGD, more than Batch GD) — requires learning rate scheduling for precise convergence.
- ❌ **Not deterministic** — results vary between runs due to random shuffling (unless you fix the random seed).

**When to use:** **Almost always.** Mini-batch GD is the default choice in practice for most machine learning and deep learning applications.

---

#### Summary Comparison

| Aspect | Batch GD | Stochastic GD | Mini-Batch GD |
|:---|:---|:---|:---|
| **Samples per update** | All $m$ | 1 | Mini-batch of $b$ (32–256) |
| **Updates per epoch** | 1 | $m$ | $m / b$ |
| **Gradient quality** | Exact | Very noisy | Moderately noisy |
| **Convergence path** | Smooth, direct | Zigzag, erratic | Balanced |
| **Speed per epoch** | Slow | Fast (but many steps) | Fast (vectorized) |
| **Memory** | High ($O(m)$) | Low ($O(1)$) | Medium ($O(b)$) |
| **Hardware utilization** | Good (vectorized) | Poor (scalar) | Best (optimized batch ops) |
| **Online learning** | ❌ | ✅ | ❌ (but can adapt) |
| **Escapes local minima** | ❌ | ✅ | Partially |
| **Default choice?** | Small data only | Rarely used alone | **Yes — standard in practice** |

![Gradient Descent with Momentum - Convergence Paths](images/05_gd_convergence_paths.svg)

*Comparison of convergence paths. Batch GD follows a smooth path, SGD zigzags noisily, and Mini-Batch GD balances both.*

**When to Use Gradient Descent (general):** Large datasets with many features (>10,000)

## Polynomial Regression

### What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression that models **non-linear relationships** between features and the target variable by adding polynomial terms (powers and interactions of the original features) to the model.

Despite fitting curves, Polynomial Regression is still a **linear model** — it is linear in the **parameters** $\theta$, not in the features $x$. We simply create new features ($x^2, x^3, \ldots$) and then fit a standard linear model on the expanded feature set.

### Equation

For a single feature $x$, polynomial regression of degree $d$:

$$\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \cdots + \theta_d x^d$$

In matrix form, we transform the feature matrix $X$ into a polynomial feature matrix $X_{poly}$ and then apply regular linear regression:

$$\hat{y} = X_{poly} \theta$$

### Visual Intuition

![Polynomial Regression Example](images/06_polynomial_regression.png)

*Polynomial regression fits a curve to data that a straight line cannot capture. Higher degrees fit more complex patterns but risk overfitting.*

### How It Works

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

### Choosing the Polynomial Degree

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

### Overfitting and Regularization

Polynomial Regression is highly prone to overfitting because:
1. **High-degree polynomials** can fit arbitrarily complex curves, including noise.
2. **Coefficient magnitudes** can become very large, leading to wild oscillations between data points.

![Underfitting vs Overfitting](images/07_overfitting.png)

*Left: Underfitting (degree too low). Right: Overfitting (degree too high). The model fits training data perfectly but generalizes poorly.*

**Solutions:**
- **Regularization** (Ridge, Lasso, Elastic Net) — constrains coefficient magnitudes. See [regularization_techniques.md](regularization_techniques.md).
- **Reduce degree** — simplest fix.
- **More training data** — harder for the model to memorize.
- **Feature scaling** — essential when using polynomial features because $x^d$ magnifies scale differences enormously.

### Python Implementation

**Using Scikit-learn:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a pipeline: polynomial features → scaling → linear regression
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

poly_model.fit(X_train, y_train)
predictions = poly_model.predict(X_test)
```

**With regularization (recommended):**

```python
from sklearn.linear_model import Ridge

poly_ridge = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

poly_ridge.fit(X_train, y_train)
```

**From scratch with NumPy:**

```python
import numpy as np

# Generate polynomial features for a single feature
def polynomial_features(X, degree):
    """Transform X into polynomial features [X, X², X³, ..., Xᵈ]."""
    X_poly = np.column_stack([X ** d for d in range(1, degree + 1)])
    return X_poly

# Example
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
X_poly = polynomial_features(X, degree=3)
# X_poly = [[1, 1, 1], [2, 4, 8], [3, 9, 27], [4, 16, 64], [5, 25, 125]]

# Add intercept and fit using Normal Equation
X_b = np.c_[np.ones(X_poly.shape[0]), X_poly]
theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
```

### Key Takeaways

| Aspect | Detail |
|:---|:---|
| **What it does** | Adds polynomial terms to model non-linear relationships |
| **Still linear?** | Yes — linear in parameters, not in features |
| **Main hyperparameter** | Polynomial degree $d$ |
| **Danger** | Overfitting increases sharply with higher degrees |
| **Must-do** | Feature scaling (polynomial terms amplify scale differences) |
| **Best practice** | Use with regularization (Ridge or Elastic Net) |
| **Feature explosion** | $n=100, d=2$ → 5,151 features — be careful with high $n$ or $d$ |

---

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

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

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

$$J(\theta) = \frac{1}{2m} \sum (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

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

$$J(\theta) = \frac{1}{2m} \sum (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

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

$$J(\theta) = \frac{1}{2m} \sum (y_i - \hat{y}_i)^2 + r \cdot \lambda \sum |\theta_j| + \frac{(1-r)}{2} \cdot \lambda \sum \theta_j^2$$

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

![Bias-Variance Tradeoff](images/10_bias_variance_tradeoff.png)

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

Where: $m$ = number of training samples, $n$ = number of features, $k$ = number of iterations/epochs.

### Training Algorithms

| Algorithm | Time Complexity | Space Complexity | Notes |
|:---|:---|:---|:---|
| **Normal Equation** | $O(n^3 + mn^2)$ | $O(mn + n^2)$ | $n^3$ for matrix inversion of $X^TX$, $mn^2$ to compute $X^TX$. Becomes impractical when $n > 10{,}000$. |
| **Batch Gradient Descent** | $O(kmn)$ | $O(mn)$ | $mn$ per iteration (full dataset), $k$ iterations. Linear in all dimensions. |
| **Stochastic GD (SGD)** | $O(kn)$ per update, $O(kmn)$ per epoch | $O(n)$ | Each update is $O(n)$. One epoch = $m$ updates. Much faster per step than Batch GD. |
| **Mini-Batch GD** | $O(kbn)$ per update, $O(kmn)$ per epoch | $O(bn)$ | $b$ = batch size. One epoch = $m/b$ updates. Best hardware utilization. |

### Regularized Models

| Algorithm | Training Complexity | Notes |
|:---|:---|:---|
| **Ridge (Normal Eq.)** | $O(n^3 + mn^2)$ | Same as Normal Equation — the $\lambda I$ addition is $O(n^2)$ and doesn't change the dominant term. |
| **Ridge (GD)** | $O(kmn)$ | Same as standard GD — the L2 gradient term adds $O(n)$ per iteration, negligible. |
| **Lasso (Coordinate Descent)** | $O(kmn)$ | $k$ = passes over all coordinates. Each coordinate update is $O(m)$. No closed-form solution. |
| **Elastic Net (Coordinate Descent)** | $O(kmn)$ | Same as Lasso — combines L1 and L2 in each coordinate update. |

### Polynomial Regression

| Step | Complexity | Notes |
|:---|:---|:---|
| **Feature expansion** | $O(m \cdot \binom{n+d}{d})$ | $d$ = polynomial degree. Creates all polynomial combinations. |
| **Training (after expansion)** | Same as LR with $p$ features | Where $p = \binom{n+d}{d}$ is the expanded feature count. Can be very large. |

Example: $n=100$, $d=2$ → $p = 5{,}151$ features. Normal Equation would require inverting a $5{,}151 \times 5{,}151$ matrix.

### Prediction (All Linear Models)

| Operation | Complexity |
|:---|:---|
| **Single prediction** | $O(n)$ — dot product of feature vector and parameter vector |
| **Batch prediction** ($m$ samples) | $O(mn)$ — matrix-vector multiplication |

### Feature Scaling

| Method | Fit (compute stats) | Transform (apply) |
|:---|:---|:---|
| **Standardization** | $O(mn)$ — compute mean and std per feature | $O(mn)$ — subtract and divide per element |
| **Min-Max** | $O(mn)$ — find min and max per feature | $O(mn)$ — subtract and divide per element |
| **Robust Scaling** | $O(mn \log m)$ — compute median and IQR (requires sorting) | $O(mn)$ — subtract and divide per element |

### Evaluation Metrics

| Metric | Complexity |
|:---|:---|
| **MSE, RMSE, MAE** | $O(m)$ — single pass over predictions |
| **R²** | $O(m)$ — two passes (compute mean, then sums) |

---

**Remember:** Linear regression is simple yet powerful. Understanding both approaches helps you choose the right tool for the problem at hand!
