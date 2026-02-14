# Introduction to Artificial Neural Networks

Artificial Neural Networks (ANNs) are computing systems loosely inspired by biological neural networks. They learn to perform tasks by adjusting internal parameters (weights and biases) based on examples, without being explicitly programmed with task-specific rules.

**Core idea:** Stack layers of simple computational units (neurons) that each compute a weighted sum of inputs, add a bias, and apply a nonlinear activation function. By chaining many such units, the network can learn arbitrarily complex mappings from inputs to outputs.

---

## From Biological to Artificial Neurons

### Biological Neurons

A biological neuron is a cell in the nervous system that processes and transmits information via electrical and chemical signals.

**Key components:**

| Component | Function |
|-----------|----------|
| **Dendrites** | Receive signals (inputs) from other neurons |
| **Cell body (soma)** | Integrates incoming signals — sums them up |
| **Axon** | Transmits the output signal to other neurons |
| **Synapses** | Connections between neurons where signal transmission occurs; each synapse has a certain "strength" (weight) |

**How it works (simplified):**

1. A neuron receives signals from thousands of other neurons through its dendrites
2. These signals are **integrated** (summed) in the cell body
3. If the total signal exceeds a **threshold**, the neuron "fires" — it sends an electrical impulse down its axon
4. This signal is transmitted across synapses to other neurons, with each synapse modulating the signal strength

**The biological analogy for artificial neurons:**

| Biological | Artificial |
|-----------|------------|
| Dendrites (inputs) | Input features $x_1, x_2, \ldots, x_n$ |
| Synaptic strength | Weights $w_1, w_2, \ldots, w_n$ |
| Cell body (integration) | Weighted sum $z = \sum w_i x_i + b$ |
| Firing threshold | Activation function $\sigma(z)$ |
| Axon output | Neuron output $\hat{y}$ |

> **Important caveat:** While ANNs were *inspired* by biology, they are a dramatic simplification. Real neurons use complex temporal dynamics, spike timing, neurotransmitter chemistry, and dendritic computation that ANNs do not model. The analogy is useful for intuition but should not be taken literally.

---

### Logical Computations with Neurons

McCulloch and Pitts (1943) showed that even a **single artificial neuron** with binary inputs/outputs can compute basic logical functions. This was one of the first demonstrations that simple neural units can perform computation.

**A binary threshold neuron** outputs 1 if the weighted sum of its inputs exceeds a threshold, and 0 otherwise:

$$y = \begin{cases} 1 & \text{if } \sum_i w_i x_i + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

**Examples of logical gates as single neurons:**

| Gate | $w_1$ | $w_2$ | $b$ (bias) | Rule |
|------|--------|--------|------------|------|
| **AND** | 1 | 1 | $-1.5$ | Fires only if both inputs are 1 |
| **OR** | 1 | 1 | $-0.5$ | Fires if at least one input is 1 |
| **NOT** | $-1$ | — | $0.5$ | Fires if input is 0 (inverts) |
| **NAND** | $-1$ | $-1$ | $1.5$ | Fires unless both inputs are 1 |

**Verification (AND gate):**

| $x_1$ | $x_2$ | $z = 1 \cdot x_1 + 1 \cdot x_2 - 1.5$ | $y$ |
|--------|--------|------------------------------------------|------|
| 0 | 0 | $-1.5$ | 0 |
| 0 | 1 | $-0.5$ | 0 |
| 1 | 0 | $-0.5$ | 0 |
| 1 | 1 | $+0.5$ | **1** |

**Why this matters:** Since NAND gates are **functionally complete** (any Boolean function can be built from NANDs alone), networks of simple neurons can compute *any* logical function. This was an early theoretical motivation for neural networks.

> **The XOR problem:** A single neuron **cannot** compute XOR ($x_1 \oplus x_2$), because XOR is not linearly separable. This limitation motivated the development of **multilayer** networks.

---

## The Perceptron

The **Perceptron** (Rosenblatt, 1957) is the simplest artificial neural network — a single layer of binary threshold neurons (called Threshold Logic Units, TLUs).

### Architecture

Each TLU computes:

$$z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b$$

$$\hat{y} = \text{step}(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

A **single-layer Perceptron** has:
- An **input layer** (just passes through the features — no computation)
- An **output layer** of one or more TLUs (each making a binary decision)

For a full layer, the computation in matrix form is:

$$\hat{\mathbf{y}} = \text{step}(\mathbf{X} \mathbf{W} + \mathbf{b})$$

Where:
- $\mathbf{X}$ is the input matrix (rows = instances, columns = features)
- $\mathbf{W}$ is the weight matrix (rows = input features, columns = output neurons)
- $\mathbf{b}$ is the bias vector
- $\text{step}(\cdot)$ is applied element-wise

### The Perceptron Learning Rule

The Perceptron learns by updating weights whenever it makes a mistake:

$$w_{i,j}^{(\text{next})} = w_{i,j} + \eta \, (y_j - \hat{y}_j) \, x_i$$

Where:
- $\eta$ is the learning rate
- $y_j$ is the true label for output neuron $j$
- $\hat{y}_j$ is the predicted label
- $x_i$ is the $i$-th input value

**Intuition:** If the prediction is correct ($y_j = \hat{y}_j$), the weight doesn't change. If the prediction is wrong, the weight is nudged in the direction that would have made it correct.

**Key properties:**
- **Convergence theorem:** If the training data is linearly separable, the Perceptron learning rule is guaranteed to converge to a solution in a finite number of steps
- **Limitation:** If the data is *not* linearly separable, the algorithm will never converge — it oscillates forever

```python
from sklearn.linear_model import Perceptron

# Scikit-Learn's Perceptron is equivalent to SGDClassifier(loss="perceptron")
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

### Limitations of the Perceptron

The **Perceptron controversy** (Minsky & Papert, 1969) demonstrated critical limitations:

| Limitation | Explanation |
|-----------|-------------|
| **Cannot solve XOR** | XOR is not linearly separable — no single hyperplane can separate the classes |
| **Linear decision boundaries only** | A single layer of TLUs can only represent linear functions |
| **Unstable on non-separable data** | Doesn't converge if classes overlap |

This led to the first "AI winter" for neural networks. The solution — **multilayer** networks — was known in theory, but an efficient training algorithm was missing until the popularization of **backpropagation** in 1986.

---

## The Multilayer Perceptron and Backpropagation

### Architecture

A **Multilayer Perceptron (MLP)** consists of:

1. **Input layer** — one neuron per input feature (passthrough, no computation)
2. **One or more hidden layers** — each a fully connected layer of neurons with nonlinear activation functions
3. **Output layer** — produces the final predictions

**Key difference from the Perceptron:** Hidden layers use **smooth, differentiable** activation functions (not the step function), which enables gradient-based training.

### Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Output layer for binary classification |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Hidden layers (zero-centered, better than sigmoid) |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | $[0, \infty)$ | Default for hidden layers — fast, avoids vanishing gradients |
| **Softmax** | $\sigma(z_j) = \frac{e^{z_j}}{\sum_k e^{z_k}}$ | $(0, 1)$, sums to 1 | Output layer for multiclass classification |

**Why nonlinear activation functions are essential:**
- Without nonlinearity, stacking layers is useless: a chain of linear transformations is just another linear transformation ($W_2 \cdot W_1 = W_{\text{combined}}$)
- Nonlinear activations allow the network to learn **nonlinear decision boundaries** and approximate any continuous function (Universal Approximation Theorem)

### Backpropagation

**Backpropagation** (Rumelhart, Hinton & Williams, 1986) is the algorithm that makes training deep networks practical. It efficiently computes gradients of the loss with respect to every weight in the network.

**The algorithm in two phases:**

**1. Forward pass:**
- Input flows through the network layer by layer
- Each neuron computes $z = \mathbf{w} \cdot \mathbf{x} + b$, then $a = \sigma(z)$
- The final output is compared to the true target using a **loss function** $\mathcal{L}$

**2. Backward pass (backpropagation):**
- Compute $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ at the output
- Apply the **chain rule** to propagate gradients backward through each layer:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{(l)}} = \frac{\partial \mathcal{L}}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}$$

- Each layer passes the gradient to the previous layer, reusing intermediate computations (this is what makes it efficient)

**3. Weight update (Gradient Descent):**

$$w_{ij} \leftarrow w_{ij} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}}$$

**Why backpropagation is efficient:**
- Without it, you'd need to compute the gradient of each weight independently — astronomically expensive
- Backprop computes **all gradients in one backward pass** by reusing intermediate results (reverse-mode automatic differentiation)
- Computational cost: roughly **2× the forward pass** (one forward + one backward)

### Common Loss Functions

| Task | Loss Function | Formula |
|------|--------------|---------|
| **Regression** | Mean Squared Error (MSE) | $\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2$ |
| **Binary classification** | Binary Cross-Entropy | $\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| **Multiclass classification** | Categorical Cross-Entropy | $\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$ |

### Gradient Descent Variants

| Variant | Batch Size | Pros | Cons |
|---------|-----------|------|------|
| **Batch GD** | Entire dataset | Stable convergence, clean gradients | Slow on large datasets, high memory |
| **Stochastic GD (SGD)** | 1 instance | Fast updates, can escape local minima | Noisy gradients, erratic convergence |
| **Mini-batch GD** | Typically 32–256 | Best of both worlds — fast and reasonably stable | Requires tuning batch size |

> **In practice**, mini-batch GD is the default. Modern optimizers (Adam, RMSProp) build on mini-batch SGD with adaptive learning rates.

---

## Building and Training MLPs with Scikit-Learn

Scikit-Learn provides `MLPRegressor` and `MLPClassifier` for quick MLP prototyping. These are best for small to medium datasets and rapid experimentation — for large-scale or GPU-accelerated work, use PyTorch or TensorFlow.

### Regression MLPs

A **regression MLP** predicts one or more continuous values.

**Output layer design:**

| Task | Output Neurons | Activation | Loss |
|------|---------------|------------|------|
| Single target | 1 | None (identity) | MSE |
| Multiple targets | 1 per target | None (identity) | MSE |
| Positive-only output | 1 | ReLU or Softplus | MSE |

> **No activation on the output layer for regression** — you want the network to output any real value, not squash it into a fixed range.

```python
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load and split data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Always scale inputs for neural networks
mlp_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 and 50 neurons
        activation='relu',             # ReLU for hidden layers
        solver='adam',                  # Adam optimizer
        max_iter=500,
        early_stopping=True,           # Stop when validation loss stops improving
        validation_fraction=0.1,       # Use 10% of training data for validation
        random_state=42
    ))
])

mlp_reg.fit(X_train, y_train)
print(f"R² score: {mlp_reg.score(X_test, y_test):.3f}")
```

**Key points for regression MLPs:**
- **Always scale inputs** — neural networks are sensitive to feature scales (use `StandardScaler` or `MinMaxScaler`)
- Use **`early_stopping=True`** to prevent overfitting — training stops when validation loss doesn't improve for `n_iter_no_change` epochs (default: 10)
- `hidden_layer_sizes=(100, 50)` means two hidden layers with 100 and 50 neurons respectively
- The `adam` solver is generally the best default for MLPs

### Classification MLPs

A **classification MLP** predicts discrete class labels.

**Output layer design:**

| Task | Output Neurons | Activation | Loss |
|------|---------------|------------|------|
| Binary classification | 1 | Sigmoid (logistic) | Binary cross-entropy |
| Multiclass (mutually exclusive) | 1 per class | Softmax | Categorical cross-entropy |
| Multilabel classification | 1 per label | Sigmoid (logistic) | Binary cross-entropy per label |

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load and split data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlp_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100,),     # One hidden layer with 100 neurons
        activation='relu',             # ReLU for hidden layers
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=42
    ))
])

mlp_clf.fit(X_train, y_train)
print(f"Accuracy: {mlp_clf.score(X_test, y_test):.3f}")
print(classification_report(y_test, mlp_clf.predict(X_test)))
```

**How scikit-learn handles classification automatically:**
- **Binary:** 1 output neuron with logistic (sigmoid) activation
- **Multiclass:** Automatically uses softmax output with as many neurons as classes
- The loss function is always **cross-entropy** (log loss)

**Accessing probabilities:**

```python
# Predicted probabilities for each class
probas = mlp_clf.predict_proba(X_test)
print(f"Class probabilities for first instance: {probas[0]}")
```

---

## Hyperparameter Tuning Guidelines

Neural networks have many hyperparameters. Here is a practical guide for choosing them.

### Number of Hidden Layers

| Depth | When to Use | Example |
|-------|-------------|---------|
| **0 hidden layers** | Linearly separable problems | Equivalent to logistic regression |
| **1 hidden layer** | Most "simple" problems — can approximate any continuous function (Universal Approximation Theorem) | Tabular data, simple patterns |
| **2 hidden layers** | More complex patterns, faster convergence than 1 wide layer for many problems | Most practical problems |
| **3+ hidden layers** | Hierarchical feature learning, complex spatial/temporal data | Images, text, speech (use deep learning frameworks) |

**Practical guidance:**
- **Start with 1–2 hidden layers** — this handles the vast majority of tabular/structured data problems
- Deep networks (3+ layers) learn **hierarchical representations**: early layers learn low-level features, later layers combine them into high-level concepts
- More layers does **not** always mean better performance — deeper networks are harder to train (vanishing gradients, longer training, more data needed)
- **Transfer learning** often removes the need for training very deep networks from scratch

> **Rule of thumb:** For most problems you encounter with scikit-learn, 1–2 hidden layers are sufficient. If you need more depth, switch to PyTorch or TensorFlow which have better tools for training deep networks (batch normalization, residual connections, GPU support).

### Number of Neurons per Hidden Layer

**Too few neurons → underfitting** (network can't capture the complexity of the data)
**Too many neurons → overfitting** (network memorizes instead of generalizing) and slower training

**Practical strategies:**

| Strategy | Description |
|----------|-------------|
| **Pyramid (shrinking)** | Each layer has fewer neurons than the previous: e.g., `(300, 200, 100)` — forces the network to learn compressed representations |
| **Constant width** | All hidden layers the same size: e.g., `(150, 150, 150)` — simpler to tune, often works just as well |
| **Stretch pants approach** | Pick a model with more layers and neurons than you need, then use regularization and early stopping to prevent overfitting |

**Sizing guidelines:**

| Dataset Size | Suggested Starting Point |
|-------------|--------------------------|
| Small (< 1K samples) | `(50,)` or `(50, 25)` |
| Medium (1K–100K) | `(100, 50)` or `(100, 100)` |
| Large (100K+) | `(300, 200, 100)` or wider |

> **The "stretch pants" approach is generally preferred today:** rather than carefully sizing each layer, use a large network and rely on **early stopping** and **regularization** to prevent overfitting. This is simpler and often gives better results than trying to find the minimum sufficient architecture.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Search over architectures
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=500, early_stopping=True, random_state=42))
])

param_grid = {
    'mlp__hidden_layer_sizes': [
        (50,), (100,), (200,),           # 1 hidden layer
        (100, 50), (100, 100),           # 2 hidden layers
        (200, 100, 50),                  # 3 hidden layers
    ],
    'mlp__alpha': [1e-4, 1e-3, 1e-2],   # L2 regularization
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best architecture: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_:.3f}")
```

### Learning Rate

The **learning rate** $\eta$ controls the step size during gradient descent weight updates.

$$w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}$$

| Learning Rate | Behavior |
|--------------|----------|
| **Too large** (e.g., 0.1+) | Training diverges — loss oscillates or explodes |
| **Too small** (e.g., 1e-5) | Training converges very slowly, may get stuck in poor local minima |
| **Just right** (typically 1e-4 to 1e-2) | Smooth, steady decrease in loss |

**Scikit-learn learning rate options:**

```python
MLPClassifier(
    solver='sgd',
    learning_rate='constant',          # Fixed learning rate
    learning_rate_init=0.01,           # Starting learning rate
)

MLPClassifier(
    solver='sgd',
    learning_rate='adaptive',          # Divides by 5 when loss stalls
    learning_rate_init=0.01,
)

# Adam (recommended default) — handles learning rate adaptation internally
MLPClassifier(
    solver='adam',
    learning_rate_init=0.001,          # Adam's default, usually works well
)
```

**Practical guidance:**
- **Use `solver='adam'`** as default — it adapts the learning rate per-parameter automatically
- If using SGD, start with `learning_rate_init=0.01` and use `learning_rate='adaptive'`
- **Learning rate schedules** (reducing the rate over time) help reach better optima — `adaptive` does this in scikit-learn, while PyTorch/TensorFlow offer more sophisticated schedulers

### Batch Size

The **batch size** determines how many training instances are used to compute each gradient update.

| Batch Size | Pros | Cons |
|-----------|------|------|
| **Small** (16–32) | Acts as regularization (noisy gradients), better generalization, less memory | Slower per epoch (more updates), noisier training |
| **Medium** (64–256) | Good balance of speed and generalization | — |
| **Large** (512+) | Faster training per epoch, more stable gradients | May generalize worse (sharp minima), needs learning rate scaling, high memory |

> **Recent research insight:** Small batch sizes often lead to **better generalization** because the noise in gradient estimates helps the optimizer find **flat minima** (which generalize better) rather than **sharp minima** (which don't).

**In scikit-learn**, the `batch_size` parameter controls this:

```python
MLPClassifier(
    solver='adam',
    batch_size=32,     # Mini-batch size (default: 'auto' = min(200, n_samples))
    max_iter=500,
)
```

**Practical guidance:**
- Start with **batch size 32** — a good default that balances speed and generalization
- If training is too slow, increase to 64 or 128
- If overfitting, try decreasing to 16
- When increasing batch size, consider increasing the learning rate proportionally (linear scaling rule)

### Other Hyperparameters

| Hyperparameter | What it Does | Recommended Default | Notes |
|---------------|-------------|---------------------|-------|
| **`alpha`** (L2 regularization) | Penalizes large weights: adds $\alpha \|\mathbf{w}\|^2$ to loss | `1e-4` | Increase if overfitting, decrease if underfitting |
| **`solver`** | Optimization algorithm | `'adam'` | Adam is best for most cases; `'sgd'` with momentum for fine control; `'lbfgs'` for small datasets |
| **`activation`** | Hidden layer activation function | `'relu'` | ReLU is the default choice; `'tanh'` if you need zero-centered outputs |
| **`early_stopping`** | Stop training when validation loss stops improving | `True` | Almost always use this to prevent overfitting |
| **`n_iter_no_change`** | Patience for early stopping | `10` | Number of epochs with no improvement before stopping |
| **`momentum`** | SGD momentum (accelerates convergence) | `0.9` | Only applies when `solver='sgd'` |
| **`beta_1`, `beta_2`** | Adam exponential decay rates | `0.9`, `0.999` | Rarely need to change these |
| **`epsilon`** | Adam numerical stability constant | `1e-8` | Almost never needs changing |
| **`shuffle`** | Shuffle samples each epoch | `True` | Always keep this on for SGD/Adam |

### Putting It All Together — A Complete Example

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import numpy as np

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=1000, early_stopping=True, random_state=42))
])

# Randomized search is more efficient than grid search for neural networks
param_distributions = {
    'mlp__hidden_layer_sizes': [
        (50,), (100,), (100, 50), (100, 100), (200, 100),
    ],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam'],
    'mlp__alpha': loguniform(1e-5, 1e-1),           # Log-uniform for regularization
    'mlp__learning_rate_init': loguniform(1e-4, 1e-1),
    'mlp__batch_size': [16, 32, 64, 128],
}

search = RandomizedSearchCV(
    pipe,
    param_distributions,
    n_iter=50,                # Try 50 random combinations
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
)

search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.3f}")
print(f"Test accuracy: {search.score(X_test, y_test):.3f}")
```

---

## Quick Reference Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Biological → Artificial** | Neurons sum weighted inputs, apply activation; synaptic strength ≈ weights |
| **Perceptron** | Single-layer, linear boundaries only, cannot solve XOR |
| **MLP** | Multiple layers with nonlinear activations → can learn any function |
| **Backpropagation** | Chain rule applied backward through network; computes all gradients efficiently |
| **Activation functions** | ReLU (hidden), sigmoid (binary output), softmax (multiclass output) |
| **Regression MLP** | No activation on output, MSE loss |
| **Classification MLP** | Softmax/sigmoid output, cross-entropy loss |
| **Scaling** | Always scale inputs (StandardScaler) — neural networks are sensitive to feature magnitude |
| **Architecture** | Start with 1–2 hidden layers, 50–200 neurons; use "stretch pants" + early stopping |
| **Learning rate** | Use Adam optimizer with `lr=0.001` as default; too high → divergence, too low → slow |
| **Batch size** | Start with 32; smaller = better generalization, larger = faster training |
| **Regularization** | L2 (`alpha`), early stopping, and dropout (in deep learning frameworks) |
