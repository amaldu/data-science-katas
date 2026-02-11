# Batch Learning vs. Online Learning

## Overview

Machine learning models need to learn from data, but **how** and **when** they learn varies fundamentally. The two main paradigms are:

- **Batch Learning (Offline Learning):** The model is trained on the entire dataset at once, then deployed. It does not learn from new data unless retrained from scratch.
- **Online Learning (Incremental Learning):** The model learns continuously, updating its parameters as each new data point (or small group of data points) arrives.

This distinction affects how you design, deploy, and maintain ML systems in production.

---

## Batch Learning (Offline Learning)

### Definition

Batch learning is a training paradigm where the model is trained on a **fixed, complete dataset** in one go. The algorithm sees the entire dataset — often multiple times (epochs) — before producing a final model. Once training is complete, the model is deployed and makes predictions without further learning.

If new data becomes available and the model needs updating, you must **retrain from scratch** on the full dataset (old data + new data), then replace the deployed model.

### How It Works

```
Step 1: Collect and store the entire training dataset
Step 2: Train the model on the full dataset (possibly for many epochs)
Step 3: Evaluate the model on a held-out test set
Step 4: Deploy the trained model to production
Step 5: The model makes predictions but does NOT learn from new data
Step 6: When new data accumulates, retrain from scratch and redeploy
```

### Training Process

In batch learning, a typical training loop looks like this:

```python
# Batch learning: train on the full dataset
for epoch in range(num_epochs):
    predictions = model.predict(X_train)       # Use ALL training data
    loss = compute_loss(predictions, y_train)   # Compute loss over ALL samples
    gradients = compute_gradients(loss)         # Average gradient over ALL samples
    model.update_parameters(gradients)          # One update per epoch
```

The model sees **every sample** before making a single parameter update (in pure batch mode) or processes the dataset in large chunks (mini-batches) over multiple epochs.

### Examples of Batch Learning

| Algorithm | How It's Batch | Time Complexity |
|:---|:---|:---|
| **Linear Regression (Normal Equation)** | Computes $\theta = (X^TX)^{-1}X^Ty$ using the entire dataset in one step | $O(n^3 + mn^2)$ |
| **Batch Gradient Descent** | Computes the gradient over all $m$ samples before each update | $O(kmn)$ per training run |
| **Random Forest** | Builds each tree using bootstrap samples of the full dataset | $O(T \cdot mn\log m)$ where $T$ = number of trees |
| **SVM** | Solves an optimization problem using the entire dataset | $O(m^2 n)$ to $O(m^3)$ depending on kernel |
| **Most scikit-learn models** | `model.fit(X_train, y_train)` trains on the full dataset at once | Varies by algorithm |

### Advantages

- **Simplicity:** Easy to implement, debug, and reason about. Train once, deploy, done.
- **Stability:** Using the full dataset produces stable, low-variance gradient estimates. Convergence is smooth and predictable.
- **Reproducibility:** Given the same data and random seed, you get the same model every time.
- **Mature tooling:** Most ML frameworks (scikit-learn, XGBoost, etc.) are designed for batch learning by default.
- **Best offline accuracy:** With enough compute, batch learning can fully optimize the model on the available data, often achieving the best possible performance on a static dataset.

### Disadvantages

- **Cannot adapt to new data:** The model is frozen after training. If the data distribution changes (concept drift), the model becomes stale and performance degrades.
- **High memory requirements:** The entire dataset must fit in memory (or be efficiently loaded). For very large datasets (terabytes), this can be prohibitive.
- **Expensive retraining:** Every time you want to incorporate new data, you retrain from scratch on the full dataset. This costs time and compute.
- **High latency for updates:** Retraining can take hours, days, or weeks for large models. During that time, the deployed model may be outdated.
- **Wasteful:** When only a small amount of new data arrives, retraining on the entire dataset (old + new) repeats work already done.

### When to Use Batch Learning

- The dataset is **static** or changes infrequently (e.g., monthly data dumps).
- The dataset is **small to medium** and fits comfortably in memory.
- You don't need real-time model updates.
- **Accuracy** is more important than adaptation speed.
- The data distribution is **stationary** (doesn't change over time).
- You want **simplicity** in your ML pipeline.

---

## Online Learning (Incremental Learning)

### Definition

Online learning is a training paradigm where the model **learns incrementally**, updating its parameters as each new data point (or small mini-batch) arrives. The model is always learning — it never stops training. After processing each sample, it can immediately make better predictions.

The key idea is that the model does not need access to the full dataset at any point. Data can be fed in as a stream, and each sample is used once (or a few times) and then discarded.

### How It Works

```
Step 1: Initialize the model with random parameters
Step 2: Receive a new data point (or small batch) from the stream
Step 3: Make a prediction with the current model
Step 4: Observe the true label (feedback)
Step 5: Compute the loss on this single sample
Step 6: Update the model parameters using this sample's gradient
Step 7: Discard the sample (no need to store it)
Step 8: Go to Step 2 — repeat indefinitely
```

### Training Process

```python
# Online learning: update after every single sample
for x_i, y_i in data_stream:              # Data arrives one at a time
    prediction = model.predict(x_i)         # Predict with current model
    loss = compute_loss(prediction, y_i)    # Loss on ONE sample
    gradient = compute_gradient(loss)        # Gradient from ONE sample
    model.update_parameters(gradient)        # Update immediately
    # x_i and y_i can now be discarded
```

### The Learning Rate in Online Learning

The **learning rate** $\alpha$ takes on special importance in online learning because it controls how much each new sample influences the model:

| Learning Rate | Effect |
|:---:|:---|
| **High $\alpha$** | Model adapts quickly to new data, but also reacts strongly to noise and outliers. Forgets old patterns fast. |
| **Low $\alpha$** | Model changes slowly, more stable, but adapts slowly to genuine changes in the data distribution. |

This creates a fundamental trade-off:

$$\text{High } \alpha \implies \text{fast adaptation, high sensitivity to noise}$$

$$\text{Low } \alpha \implies \text{slow adaptation, high stability}$$

**In practice:** Use a decaying learning rate or adaptive methods (Adam, AdaGrad) that automatically adjust.

### Examples of Online Learning

| Algorithm | How It's Online | Time Complexity (per update) |
|:---|:---|:---|
| **Stochastic Gradient Descent (SGD)** | Updates parameters after each sample — the core engine of online learning | $O(n)$ |
| **Perceptron** | Classic online algorithm: updates weights only when it makes a mistake | $O(n)$ |
| **Online SVMs** | Incremental versions of SVMs that update with each new sample | $O(n_s \cdot n)$ where $n_s$ = support vectors |
| **Bayesian Online Learning** | Updates posterior distribution incrementally using Bayes' theorem | $O(n^2)$ to $O(n^3)$ depending on model |
| **Bandit algorithms** | Learn which action to take by observing rewards one at a time | $O(K)$ where $K$ = number of arms |
| **Vowpal Wabbit** | Dedicated online learning framework designed for large-scale streaming data | $O(n)$ per sample |

### Advantages

- **Adapts to changing data:** Online learning naturally handles **concept drift** — when the statistical properties of the data change over time (e.g., user preferences shifting, market trends evolving).
- **Low memory footprint:** Only one sample (or a small batch) needs to be in memory at any time. Can handle datasets that are too large to fit in memory.
- **Real-time updates:** The model improves continuously. New data is incorporated instantly without waiting for a full retraining cycle.
- **Efficient with new data:** Each new sample is used immediately to improve the model. No need to retrain on the entire historical dataset.
- **Works with data streams:** Essential for applications where data arrives continuously (IoT sensors, financial markets, user activity logs, social media feeds).
- **No storage requirements:** Data can be processed and discarded — useful when storing the full dataset is impractical or prohibited (e.g., privacy regulations).

### Disadvantages

- **Noisy updates:** Each update is based on a single sample, so the gradient estimate is very noisy. The model may oscillate instead of converging smoothly.
- **Sensitive to outliers:** A single anomalous data point can push the model parameters significantly in the wrong direction.
- **Catastrophic forgetting:** The model can gradually forget old patterns as it adapts to new data, especially with a high learning rate.
- **Harder to evaluate:** Traditional train/test split evaluation doesn't work well. You need online evaluation methods like **prequential evaluation** (test-then-train) or sliding-window metrics.
- **Order dependence:** The sequence in which data arrives can affect the final model. Different orderings may produce different results.
- **Complex infrastructure:** Deploying online learning in production requires streaming pipelines, monitoring, and safeguards against bad data — significantly more engineering than batch learning.
- **Harder to reproduce:** Because the model changes with every sample and depends on data order, reproducing exact results is difficult.

### When to Use Online Learning

- Data arrives as a **continuous stream** (real-time applications).
- The dataset is **too large** to fit in memory or to retrain on regularly.
- The data distribution **changes over time** (concept drift, non-stationarity).
- You need **real-time model updates** (fraud detection, recommendation systems, ad click prediction).
- **Storage is limited** or data cannot be stored (privacy, regulatory constraints).
- **Low latency** is critical — you can't wait hours or days for batch retraining.

---

## Out-of-Core Learning

**Out-of-core learning** is a related concept that bridges batch and online learning. It refers to training on a dataset that is **too large to fit in memory** by loading and processing it in chunks.

```python
# Out-of-core learning: process data in chunks
for chunk in load_data_in_chunks(filepath, chunk_size=10000):
    model.partial_fit(chunk.X, chunk.y)  # Incremental update
```

| Aspect | Batch | Out-of-Core | Online |
|:---|:---|:---|:---|
| **Data in memory** | All at once | One chunk at a time | One sample at a time |
| **Data source** | Static file | Static file (large) | Live stream |
| **Passes over data** | Multiple epochs | Usually 1 pass | 1 pass (data discarded) |
| **Goal** | Best model on fixed data | Handle large static data | Continuous adaptation |

**scikit-learn** supports out-of-core learning via the `partial_fit()` method on certain estimators:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()

for X_chunk, y_chunk in data_chunks:
    model.partial_fit(X_chunk, y_chunk, classes=[0, 1])
```

Estimators that support `partial_fit()`: `SGDClassifier`, `SGDRegressor`, `Perceptron`, `PassiveAggressiveClassifier`, `MiniBatchKMeans`, `MultinomialNB`, `BernoulliNB`.

---

## Concept Drift

One of the main reasons to use online learning is to handle **concept drift** — when the relationship between input features and the target variable changes over time.

### Types of Concept Drift

| Type | Description | Example |
|:---|:---|:---|
| **Sudden drift** | The distribution changes abruptly at a single point in time | A new regulation changes customer behavior overnight |
| **Gradual drift** | The old and new distributions overlap for a period, with the new one slowly dominating | Fashion trends gradually shifting over months |
| **Incremental drift** | The distribution shifts slowly and continuously | Inflation causing prices to rise gradually |
| **Recurring drift** | The distribution alternates between known patterns | Seasonal shopping patterns (holiday spikes) |

### How Online Learning Handles Drift

- **High learning rate:** Gives more weight to recent samples, quickly adapting to the new distribution. But may overreact to noise.
- **Sliding window:** Only train on the most recent $N$ samples, discarding older data that may no longer be relevant.
- **Weighted samples:** Assign higher weights to recent samples, lower weights to older ones (exponential decay).
- **Drift detection:** Monitor model performance metrics. When performance drops significantly, increase the learning rate or reset the model.

```python
# Sliding window approach
from collections import deque

window = deque(maxlen=1000)  # Keep last 1000 samples

for x_i, y_i in data_stream:
    window.append((x_i, y_i))
    # Retrain or update model using only the window
    model.partial_fit([x_i], [y_i])
```

---

## Side-by-Side Comparison

| Aspect | Batch Learning | Online Learning |
|:---|:---|:---|
| **Training data** | Fixed, complete dataset | Streaming, one sample at a time |
| **Update frequency** | After seeing all data (per epoch) | After each sample (or small batch) |
| **Time per update** | $O(mn)$ — processes all $m$ samples | $O(n)$ — processes 1 sample |
| **Total training time** | $O(kmn)$ for $k$ epochs | $O(n)$ per new sample (continuous) |
| **Memory usage** | High — $O(mn)$ (full dataset in memory) | Low — $O(n)$ (one sample at a time) |
| **Adaptability** | Cannot adapt without retraining | Adapts continuously |
| **Concept drift** | Requires periodic retraining | Handles naturally |
| **Convergence** | Stable, smooth | Noisy, may oscillate |
| **Accuracy on static data** | Generally higher | May be slightly lower |
| **Latency to deploy updates** | Hours to days (retraining) | Seconds (immediate) |
| **Infrastructure complexity** | Simple (train → deploy) | Complex (streaming pipeline, monitoring) |
| **Reproducibility** | High (deterministic) | Low (order-dependent) |
| **Outlier sensitivity** | Low (averaged over all data) | High (one outlier = one bad update) |
| **Storage requirements** | Must store full dataset | Can discard data after use |
| **Typical algorithms** | Normal Eq., Random Forest, SVM, XGBoost | SGD, Perceptron, Bandits, Vowpal Wabbit |
| **scikit-learn API** | `model.fit(X, y)` | `model.partial_fit(X, y)` |

---

## Practical Decision Guide

```
Is your data static and small enough to fit in memory?
├── YES → Use Batch Learning
│         Simple, reproducible, best accuracy on fixed data.
│
└── NO → Does the data distribution change over time?
         ├── YES → Use Online Learning
         │         Adapts to concept drift, low memory, real-time updates.
         │
         └── NO → Use Out-of-Core Learning
                   Process large static data in chunks with partial_fit().
```

### Real-World Use Cases

| Use Case | Paradigm | Why |
|:---|:---|:---|
| Medical diagnosis model trained on hospital records | **Batch** | Static dataset, high accuracy needed, no real-time requirements |
| Spam filter that adapts to new spam patterns | **Online** | Spammers constantly evolve tactics (concept drift) |
| Image classification (cats vs. dogs) | **Batch** | Static dataset, distribution doesn't change |
| Stock price prediction | **Online** | Market dynamics change continuously |
| Recommendation system (Netflix, Spotify) | **Online** | User preferences shift over time |
| Fraud detection in credit card transactions | **Online** | Fraud patterns evolve, real-time decisions needed |
| Weather forecasting model trained on historical data | **Batch** | Large static dataset, periodic retraining is acceptable |
| Ad click-through rate prediction | **Online** | User behavior changes, millions of events per second |
| Training a large language model (LLM) | **Batch** | Massive static corpus, trained once (very expensive) |

---

## Key Takeaways

1. **Batch learning** trains on the full dataset at once. Simple, stable, and accurate — but cannot adapt to new data without retraining.
2. **Online learning** updates incrementally, one sample at a time. Adapts to change and works with streams — but noisy and complex to manage.
3. The choice depends on: **data size**, **data stationarity**, **latency requirements**, and **infrastructure capacity**.
4. **Mini-batch gradient descent** bridges both worlds — it's the default in deep learning and can be used in both batch and online settings.
5. **Concept drift** is the main driver for choosing online learning — if your data distribution changes, batch models go stale.
6. In practice, many production systems use a **hybrid approach**: train a batch model on historical data, then fine-tune with online learning as new data streams in.
