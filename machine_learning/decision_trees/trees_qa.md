# Decision Trees — Questions & Answers

---

## 1. What is the approximate depth of a Decision Tree trained (without restrictions) on a training set with 1 million instances?

**Approximately 20 levels deep.**

A binary Decision Tree (like CART) splits the data into two subsets at each node. In the best case (perfectly balanced tree), the depth is:

$$\text{depth} = \lceil \log_2(m) \rceil$$

For $m = 1{,}000{,}000$:

$$\log_2(1{,}000{,}000) = \frac{\ln(1{,}000{,}000)}{\ln(2)} \approx \frac{13.82}{0.693} \approx 19.93$$

So the depth is approximately **20**.

**In practice**, an unrestricted tree may be deeper than this because:
- Splits are rarely perfectly balanced — some branches go deeper than others
- The tree keeps splitting until each leaf is pure (single class) or has `min_samples_split` instances
- In the worst case (completely unbalanced), the depth could approach $m$, but this is pathological

**However**, the $O(\log_2 m)$ estimate holds well as a typical depth for reasonably distributed data, making Decision Tree predictions very fast even for large training sets.

---

## 2. Is a node's Gini impurity generally lower or greater than its parent's? Is it generally lower/greater, or always lower/greater?

**A node's Gini impurity is *generally* lower or equal to its parent's — but not *always*.**

**Why generally lower:**

The CART algorithm selects the split that minimizes the **weighted** impurity of the two child nodes:

$$J = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$

This weighted average is guaranteed to be **less than or equal to** the parent's impurity (because the algorithm only makes a split if it reduces the weighted impurity). This is a mathematical property of splitting a set into subsets by any criterion — the weighted impurity cannot increase.

**Why not always lower for individual children:**

While the *weighted average* of the children's impurities is always ≤ the parent's impurity, an **individual** child node can have a *higher* Gini impurity than the parent.

**Example:**
- Parent: 100 samples — 40 class A, 60 class B → $G = 1 - (0.4^2 + 0.6^2) = 0.48$
- After a split:
  - Left child: 30 samples — 15 A, 15 B → $G_{\text{left}} = 0.50$ (higher than parent!)
  - Right child: 70 samples — 25 A, 45 B → $G_{\text{right}} = 0.459$
- Weighted average: $\frac{30}{100}(0.50) + \frac{70}{100}(0.459) = 0.15 + 0.321 = 0.471 < 0.48$ ✓

The left child has Gini 0.50 > 0.48 (higher than parent), but the weighted average decreased. This is a valid split because the **overall** impurity was reduced.

**Summary:**
- **Weighted average** of children's impurities: **always** ≤ parent's impurity
- **Individual** child impurity: **generally** lower, but **can be higher** than the parent's

---

## 3. If a Decision Tree is overfitting the training set, is it a good idea to try decreasing max_depth?

**Yes, decreasing `max_depth` is one of the best ways to reduce overfitting.**

An overfitting Decision Tree is too complex — it has learned the noise and peculiarities of the training data rather than the underlying pattern. Reducing `max_depth` is a form of **regularization** that constrains the tree's complexity.

**Why it works:**
- A shallower tree has fewer splits, fewer leaves, and makes coarser (simpler) decisions
- Each leaf covers more training instances, averaging out noise
- The tree can no longer memorize individual instances

**Other regularization options to try alongside or instead of `max_depth`:**

| Hyperparameter | What to do | Effect |
|---------------|-----------|--------|
| `max_depth` | Decrease | Limits tree depth directly |
| `min_samples_split` | Increase | Requires more samples before allowing a split |
| `min_samples_leaf` | Increase | Requires more samples in each leaf |
| `max_leaf_nodes` | Decrease | Limits total number of leaves |
| `ccp_alpha` | Increase | Prunes subtrees that add little value |
| `max_features` | Decrease | Considers fewer features per split (adds randomness) |

**Practical approach:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_leaf': [1, 5, 10, 20],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy'
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

---

## 4. If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?

**No, scaling the input features will NOT help with underfitting. It will have no effect at all.**

**Why scaling doesn't matter for Decision Trees:**

Decision Trees make splits based on **thresholds** on individual features (e.g., "Is feature $k \leq t$?"). The split decision depends only on the **rank order** of feature values, not their magnitude.

If you scale a feature (e.g., multiply by 100 or standardize to zero mean and unit variance), the same instances end up on the same side of the split — only the threshold value changes, not the split itself.

**Example:**
- Original: "Is income ≤ 50,000?"
- After standardization: "Is income_scaled ≤ 0.35?"
- Same split, same tree structure, same performance.

**What to do instead if a Decision Tree is underfitting:**

| Action | Why |
|--------|-----|
| **Increase `max_depth`** (or set to `None`) | Let the tree grow deeper to capture more complex patterns |
| **Decrease `min_samples_split`** | Allow splits on smaller groups |
| **Decrease `min_samples_leaf`** | Allow smaller leaves |
| **Increase `max_leaf_nodes`** (or set to `None`) | Allow more leaf regions |
| **Decrease `ccp_alpha`** (or set to 0) | Reduce post-pruning |
| **Engineer better features** | Give the tree more informative signals to split on |
| **Consider a more powerful model** | Ensemble methods, SVMs, neural networks |

> **Note:** While scaling doesn't help Decision Trees, it **does** help many other models (SVM, k-NN, logistic regression, neural networks). So if you switch to a different model to address underfitting, remember to scale.

---

## 5. If it takes one hour to train a Decision Tree on a training set containing 1 million instances, roughly how much time will it take to train another Decision Tree on a training set containing 10 million instances?

**Approximately 11.7 hours.**

The training complexity of a Decision Tree is $O(n \times m \log m)$, where $m$ is the number of instances and $n$ is the number of features.

Assuming the number of features stays the same, the training time scales as $O(m \log m)$.

**Calculation:**

$$\frac{T_2}{T_1} = \frac{m_2 \log(m_2)}{m_1 \log(m_1)} = \frac{10^7 \times \log(10^7)}{10^6 \times \log(10^6)}$$

Using natural logarithm (any base gives the same ratio):

$$= \frac{10^7 \times 16.12}{10^6 \times 13.82} = \frac{10 \times 16.12}{13.82} \approx \frac{161.2}{13.82} \approx 11.67$$

So:

$$T_2 \approx 11.67 \times 1 \text{ hour} \approx 11 \text{ hours } 40 \text{ minutes}$$

**Key insight:** The training time grows slightly **faster than linearly** with the number of instances due to the $\log m$ factor. A 10x increase in data leads to roughly a 11.7x increase in training time (not just 10x).

---

## 6. If your training set contains 100,000 instances, will setting presort=True speed up training?

**No, with 100,000 instances, `presort=True` will likely *slow down* training.**

> **Note:** The `presort` parameter has been **deprecated** since scikit-learn 0.24 and removed in later versions. This question is about the concept.

**What `presort=True` does:**

Pre-sorts the data on all features before building the tree. At each node, instead of sorting the features to find the best split, the algorithm uses the pre-sorted indices — skipping the sorting step at each node.

**Why it *doesn't* help for large datasets:**

| Aspect | `presort=False` (default) | `presort=True` |
|--------|--------------------------|----------------|
| **Sorting at each node** | Sorts only the subset of data at the current node | Uses pre-sorted indices but must manage which instances are in the current node |
| **Memory** | Low | High — stores sorted indices for all features |
| **Overhead** | Minimal | Significant — maintaining sorted-index lookups at each node |
| **Small datasets (< ~few thousand)** | Sorting is fast anyway | May slightly speed up |
| **Large datasets (> ~10,000)** | Sorting subsets is efficient | The overhead of managing pre-sorted indices **exceeds** the sorting savings |

**The trade-off:**
- **Pre-sorting costs** $O(n \times m \log m)$ upfront
- At each node, the algorithm still needs to figure out which pre-sorted instances belong to the current node, which has its own overhead
- For large $m$, this bookkeeping overhead is **more expensive** than simply sorting the smaller subsets at each node

**Rule of thumb:**
- **< 1,000 instances:** `presort=True` *might* help marginally
- **> 10,000 instances:** `presort=True` will almost certainly **slow down** training
- **100,000 instances:** Definitely **do not** pre-sort

**In modern scikit-learn:** This parameter was removed because the default behavior (no pre-sorting) is almost always better. The maintainers found that `presort=True` was essentially never beneficial in practice.

---

## 7. Train and fine-tune a Decision Tree for the moons dataset

### a. Generate the moons dataset

```python
import numpy as np
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
```

### b. Split into training and test sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} instances")
print(f"Test set: {X_test.shape[0]} instances")
```

### c. Grid search with cross-validation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_leaf_nodes': [5, 10, 15, 20, 25, 30, 50, 75, 100],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10],
}

tree_clf = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    tree_clf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
```

### d. Train on full training set and evaluate

```python
from sklearn.metrics import accuracy_score

# The best estimator is already trained on the full training set
best_tree = grid_search.best_estimator_

# Evaluate on the test set
y_pred = best_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {test_accuracy:.4f}")
# Expected: roughly 85% to 87%

print(f"Tree depth: {best_tree.get_depth()}")
print(f"Number of leaves: {best_tree.get_n_leaves()}")
```

**Expected results:**
- The best `max_leaf_nodes` is typically around 15–30 for this dataset
- Test accuracy should be in the range of **85–87%**
- The moons dataset with `noise=0.4` has significant overlap between classes, so ~85–87% is near the achievable maximum for a single Decision Tree

---

## 8. Grow a forest

### a. Generate 1,000 subsets of the training set

```python
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances_per_subset = 100

shuffle_split = ShuffleSplit(
    n_splits=n_trees,
    train_size=n_instances_per_subset,
    random_state=42
)

# Generate the subset indices
subsets = []
for train_index, _ in shuffle_split.split(X_train):
    subsets.append(train_index)

print(f"Number of subsets: {len(subsets)}")
print(f"Instances per subset: {len(subsets[0])}")
```

### b. Train one Decision Tree on each subset and evaluate

```python
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import numpy as np

# Use the best hyperparameters from the grid search above
best_params = grid_search.best_params_

# Train 1,000 trees
forest = []
individual_accuracies = []

for i, subset_idx in enumerate(subsets):
    tree = DecisionTreeClassifier(**best_params, random_state=42)
    tree.fit(X_train[subset_idx], y_train[subset_idx])
    forest.append(tree)
    
    y_pred_i = tree.predict(X_test)
    acc_i = accuracy_score(y_test, y_pred_i)
    individual_accuracies.append(acc_i)

print(f"Mean individual tree accuracy: {np.mean(individual_accuracies):.4f}")
print(f"Std of individual accuracies: {np.std(individual_accuracies):.4f}")
print(f"Min accuracy: {np.min(individual_accuracies):.4f}")
print(f"Max accuracy: {np.max(individual_accuracies):.4f}")
# Expected: ~80% mean accuracy (worse than the single tree trained on full data)
```

### c. Majority-vote predictions

```python
from scipy.stats import mode

# Collect predictions from all 1,000 trees
all_predictions = np.array([tree.predict(X_test) for tree in forest])
# Shape: (1000, n_test_instances)

print(f"Predictions matrix shape: {all_predictions.shape}")

# Majority vote: for each test instance, take the most frequent prediction
majority_votes, _ = mode(all_predictions, axis=0)
y_pred_forest = majority_votes.ravel()
```

### d. Evaluate the forest predictions

```python
forest_accuracy = accuracy_score(y_test, y_pred_forest)

print(f"\n=== Results Comparison ===")
print(f"Single tree (grid search best): {test_accuracy:.4f}")
print(f"Mean individual tree accuracy:  {np.mean(individual_accuracies):.4f}")
print(f"Forest (majority vote):         {forest_accuracy:.4f}")
print(f"Improvement over single tree:   {(forest_accuracy - test_accuracy) * 100:.2f}%")
```

**Expected results:**
- **Single tree (trained on full training set):** ~85–87% accuracy
- **Individual trees (trained on 100 instances each):** ~80% accuracy on average
- **Forest (majority vote of 1,000 trees):** ~86–88% accuracy — typically **0.5 to 1.5% better** than the single tree

**Why does the forest work?**

This is the core idea behind **Random Forest** classifiers:

1. **Each tree is a "weak learner"** — trained on a small, random subset of the data, so individually they perform worse
2. **But their errors are diverse** — because each tree sees different data, they make different mistakes
3. **Majority voting cancels out individual errors** — when most trees agree, the prediction is likely correct. Individual errors get "outvoted" by the majority

This is an instance of the **wisdom of crowds** effect. The ensemble is stronger than any individual member because:
- Individual trees have **high variance** (unstable predictions)
- Averaging/voting across many trees **reduces variance** without increasing bias significantly
- The more diverse the trees (different training data, different features), the better the ensemble

**What a real Random Forest adds on top of this:**

| Feature | Our manual forest | `RandomForestClassifier` |
|---------|------------------|--------------------------|
| Random subsets of data | ✓ (via ShuffleSplit) | ✓ (bootstrap sampling) |
| Random subsets of features | ✗ | ✓ (`max_features`) — adds more diversity |
| Out-of-bag evaluation | ✗ | ✓ (`oob_score=True`) |
| Parallel training | ✗ | ✓ (`n_jobs=-1`) |
| Optimized implementation | ✗ | ✓ |

```python
# The "real" way — much simpler!
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=1000,
    max_leaf_nodes=grid_search.best_params_.get('max_leaf_nodes', 20),
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
print(f"Scikit-Learn Random Forest accuracy: {rf_accuracy:.4f}")
```
