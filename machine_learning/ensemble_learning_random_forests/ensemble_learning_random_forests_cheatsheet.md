# Ensemble Learning and Random Forests Cheatsheet

## Table of Contents

- [Voting Classifiers](#voting-classifiers)
- [Bagging and Pasting](#bagging-and-pasting)
- [Bagging and Pasting in Scikit-Learn](#bagging-and-pasting-in-scikit-learn)
- [Out-of-Bag Evaluation](#out-of-bag-evaluation)
- [Random Patches and Random Subspaces](#random-patches-and-random-subspaces)
- [Random Forests](#random-forests)
- [Extra-Trees](#extra-trees)
- [Feature Importance](#feature-importance)
- [Boosting](#boosting)
  - [AdaBoost](#adaboost)
  - [Gradient Boosting](#gradient-boosting)
- [Stacking](#stacking)

---

## Voting Classifiers

A **Voting Classifier** aggregates the predictions of multiple diverse classifiers (e.g., Logistic Regression, SVM, Random Forest) and predicts the class that receives the most votes.

### Hard Voting

Each classifier casts a single vote for a class, and the majority wins (mode of the predictions).

### Soft Voting

Each classifier provides a probability vector for each class. The probabilities are averaged across classifiers, and the class with the highest average probability is selected. Soft voting typically outperforms hard voting because it gives more weight to highly confident predictions.

### Pros

- Often achieves higher accuracy than any single constituent classifier (wisdom of the crowd).
- Simple to implement and conceptually intuitive.
- Works with any combination of heterogeneous classifiers.

### Cons

- All classifiers must be trained and run at prediction time, increasing computational cost.
- Only effective when individual classifiers are diverse and reasonably accurate (better than random guessing).
- Soft voting requires all classifiers to support `predict_proba`.

### Usage

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

clf1 = LogisticRegression()
clf2 = SVC(probability=True)
clf3 = DecisionTreeClassifier()

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('svc', clf2), ('dt', clf3)],
    voting='soft'  # or 'hard'
)
voting_clf.fit(X_train, y_train)
```

### Common Pitfalls

- Using classifiers that are too similar (e.g., three Decision Trees with the same hyperparameters) provides little diversity and minimal improvement.
- Forgetting to set `probability=True` on SVC when using soft voting.
- Assuming that combining weak classifiers will always improve results — each learner must perform better than random chance.

---

## Bagging and Pasting

Both **Bagging** (Bootstrap Aggregating) and **Pasting** train multiple instances of the same algorithm on different random subsets of the training data, then aggregate their predictions.

| Aspect | Bagging | Pasting |
|--------|---------|---------|
| Sampling | **With** replacement (bootstrap) | **Without** replacement |
| Diversity | Higher (due to bootstrap overlap variation) | Lower |
| Bias | Slightly higher | Slightly lower |
| Variance | Significantly lower | Lower (but less than bagging) |

### How Aggregation Works

- **Classification**: majority vote (statistical mode).
- **Regression**: average of predictions.

### Pros

- Dramatically reduces variance and overfitting compared to a single model.
- Bagging can be trained in parallel across multiple CPU cores or servers.
- Each predictor sees a different subset, introducing beneficial diversity.

### Cons

- Increases overall training time proportional to the number of estimators (mitigated by parallelism).
- May increase bias slightly (bagging) compared to a single well-tuned model.
- Less effective on low-variance models (e.g., Linear Regression).

### Usage

Best applied to high-variance, low-bias models such as Decision Trees.

### Common Pitfalls

- Using too few estimators, which does not sufficiently reduce variance.
- Applying bagging to already-low-variance models where it adds cost without benefit.
- Confusing bagging (with replacement) and pasting (without replacement).

---

## Bagging and Pasting in Scikit-Learn

Scikit-Learn provides `BaggingClassifier` and `BaggingRegressor` for both bagging and pasting.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,        # size of each subset
    bootstrap=True,         # True = bagging, False = pasting
    n_jobs=-1,              # use all available CPU cores
    random_state=42
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of base estimators in the ensemble |
| `max_samples` | Number (or fraction) of samples drawn for each estimator |
| `bootstrap` | `True` for bagging, `False` for pasting |
| `n_jobs` | Number of parallel jobs (`-1` = all cores) |
| `oob_score` | Whether to use out-of-bag samples for evaluation |

### Common Pitfalls

- Setting `max_samples` too low leads to underfitting; too high reduces diversity.
- Forgetting to set `n_jobs=-1` and missing out on parallelism.

---

## Out-of-Bag Evaluation

In bagging, each predictor is trained on roughly **63%** of the training instances (due to bootstrap sampling). The remaining ~37% that a given predictor never saw are its **out-of-bag (OOB) instances**.

Since each predictor has its own OOB set, we can evaluate each predictor on its OOB instances and average across all predictors. This provides a free validation estimate **without needing a separate validation set**.

```python
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)           # OOB accuracy
print(bag_clf.oob_decision_function_)  # OOB class probabilities per instance
```

### Pros

- No need to sacrifice data for a validation set — use all available data for training.
- Provides an unbiased estimate of generalization performance.
- Automatically available as a by-product of bagging.

### Cons

- Only available when `bootstrap=True` (bagging, not pasting).
- Slightly more computation during training to track OOB evaluations.

### Common Pitfalls

- Using OOB score with `bootstrap=False` — this will raise an error since there are no OOB samples without replacement.
- Treating OOB score as a perfect substitute for cross-validation on very small datasets.

---

## Random Patches and Random Subspaces

`BaggingClassifier` supports sampling not just rows but also **features**, adding further diversity.

| Method | Samples (rows) | Features (columns) |
|--------|:-:|:-:|
| **Bagging / Pasting** | Sampled | All |
| **Random Subspaces** | All | Sampled |
| **Random Patches** | Sampled | Sampled |

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `max_features` | Number (or fraction) of features drawn for each estimator |
| `bootstrap_features` | Whether to sample features with replacement |

```python
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.8,
    bootstrap=True,
    max_features=0.5,          # each estimator sees 50% of features
    bootstrap_features=True,
    n_jobs=-1,
    random_state=42
)
```

### Pros

- Particularly useful for high-dimensional datasets (many features).
- Increases predictor diversity, further reducing variance.
- Random Subspaces keeps all training instances, useful when data is scarce.

### Cons

- Each individual predictor becomes weaker due to seeing fewer features.
- More hyperparameters to tune.

### Common Pitfalls

- Sampling too few features so that individual learners cannot capture meaningful patterns.
- Not understanding the distinction between sampling instances and features.

---

## Random Forests

A **Random Forest** is an ensemble of Decision Trees trained via bagging (typically `max_samples` = size of training set) with an additional layer of randomness: at each node, only a random subset of features is considered for splitting.

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_leaf_nodes=16,
    n_jobs=-1,
    random_state=42
)
rf_clf.fit(X_train, y_train)
```

### Pros

- One of the most powerful and versatile ML algorithms out of the box.
- Handles both classification and regression.
- Naturally resistant to overfitting (with enough trees).
- Handles missing values and maintains accuracy for non-linear relationships.
- Requires relatively little hyperparameter tuning.

### Cons

- Less interpretable than a single Decision Tree.
- Can be slow for real-time predictions with very large ensembles.
- Tends to struggle with very high-dimensional sparse data (e.g., text) compared to linear models.
- Memory-intensive for large forests.

### Usage

Ideal for tabular data, medium-sized datasets, when you need a strong baseline with minimal tuning.

### Common Pitfalls

- Using too few trees (`n_estimators` too low): accuracy has not yet converged.
- Not setting `max_depth` or `max_leaf_nodes` for very deep forests on noisy data.
- Expecting Random Forests to extrapolate well beyond training data range (they cannot).
- Ignoring class imbalance — use `class_weight='balanced'` or resampling.

---

## Extra-Trees

**Extremely Randomized Trees (Extra-Trees)** push randomness further: instead of searching for the *best* threshold at each feature split, they use a **random threshold** for each feature.

```python
from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(
    n_estimators=500,
    n_jobs=-1,
    random_state=42
)
et_clf.fit(X_train, y_train)
```

### Pros

- **Faster training** than Random Forests because finding the optimal threshold is the most computationally expensive part of growing a tree.
- Even lower variance than Random Forests due to extra randomization.
- Trades a slightly higher bias for a significantly lower variance.

### Cons

- Slightly higher bias may reduce performance on some datasets.
- Individual trees are less accurate (though the ensemble can compensate).

### Usage

Use when training speed is critical or when you suspect overfitting with standard Random Forests. Often worth trying alongside Random Forests and comparing via cross-validation.

### Common Pitfalls

- Assuming Extra-Trees always outperform Random Forests — it depends on the dataset.
- Not comparing both approaches on your specific problem.

---

## Feature Importance

Random Forests (and tree-based ensembles in general) provide a natural measure of **feature importance**: how much each feature contributes to reducing impurity across all trees.

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rf_clf.fit(X_train, y_train)

for name, score in zip(feature_names, rf_clf.feature_importances_):
    print(f"{name}: {score:.4f}")
```

### Pros

- Quick, automatic insight into which features matter most.
- Useful for feature selection and data understanding.
- Impurity-based importances are computed during training (no extra cost).

### Cons

- Impurity-based importances can be biased toward high-cardinality features.
- Does not capture feature interactions directly.
- Correlated features share importance, making each appear less important individually.

### Common Pitfalls

- Relying solely on impurity-based importance for high-cardinality or correlated features — consider **permutation importance** as an alternative.
- Interpreting importance as causation rather than predictive relevance.

---

## Boosting

**Boosting** trains predictors sequentially, where each new predictor focuses on correcting the errors of its predecessors. Unlike bagging, boosting **cannot** be easily parallelized because each step depends on the previous one.

### AdaBoost

**Adaptive Boosting** adjusts instance weights: misclassified instances receive higher weights so subsequent classifiers focus more on hard cases.

#### Algorithm (High Level)

1. Train a base classifier on the training data.
2. Compute its weighted error rate.
3. Compute the classifier's weight (more accurate classifiers get higher weights).
4. Update instance weights: increase weights of misclassified instances.
5. Train the next classifier on the reweighted data.
6. Repeat for `n_estimators` iterations.
7. Final prediction: weighted majority vote.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # "decision stump"
    n_estimators=200,
    algorithm='SAMME',       # or 'SAMME.R' for probabilities
    learning_rate=0.5,
    random_state=42
)
ada_clf.fit(X_train, y_train)
```

#### Pros

- Can achieve very high accuracy by focusing on hard examples.
- Simple to implement with weak learners (e.g., decision stumps).
- Naturally performs feature selection.

#### Cons

- Sequential training — cannot be parallelized.
- Sensitive to noisy data and outliers (they get increasingly higher weights).
- Prone to overfitting if `n_estimators` is too high or the base learner is too complex.

#### Common Pitfalls

- Using a complex base estimator (e.g., deep trees) instead of a weak learner — leads to rapid overfitting.
- Setting `learning_rate` too high, causing oscillation, or too low, requiring excessive estimators.
- Not handling outliers in the data before training.

---

### Gradient Boosting

**Gradient Boosting** builds trees sequentially, but instead of adjusting instance weights, each new tree fits the **residual errors** (negative gradient of the loss function) of the combined ensemble so far.

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
gbrt.fit(X_train, y_train)
```

#### With Early Stopping

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

gbrt = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=500,
    learning_rate=0.1,
    n_iter_no_change=10,     # early stopping patience
    validation_fraction=0.1, # fraction for early stopping
    random_state=42
)
gbrt.fit(X_train, y_train)
print(f"Stopped at {gbrt.n_estimators_} trees")
```

#### Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `learning_rate` | Shrinks the contribution of each tree. Lower values need more trees but generalize better |
| `n_estimators` | Number of boosting stages |
| `max_depth` | Depth of each tree (typically 2–5) |
| `subsample` | Fraction of samples used per tree (<1.0 = **Stochastic Gradient Boosting**) |

#### Pros

- Often the best-performing algorithm on tabular data (especially with XGBoost, LightGBM, CatBoost).
- Flexible — supports various loss functions.
- Regularization via learning rate, tree constraints, and early stopping.

#### Cons

- Sequential training — inherently slower than bagging-based methods.
- More hyperparameters to tune than Random Forests.
- Sensitive to noisy data without proper regularization.

#### Common Pitfalls

- Not using early stopping, leading to overfitting with too many trees.
- Setting `learning_rate` too high — fast training but poor generalization.
- Forgetting to scale or clean data; Gradient Boosting is less forgiving of messy inputs than Random Forests.
- Ignoring `subsample` for stochastic boosting, which often improves generalization and speed.

---

## Stacking

**Stacking** (Stacked Generalization) trains a **meta-learner** (blender) to combine the predictions of several base classifiers, instead of using simple voting or averaging.

### How It Works

1. Split the training set into two subsets.
2. Train several base classifiers (layer 1) on the first subset.
3. Use the base classifiers to make predictions on the second subset (hold-out set).
4. Train a meta-learner (layer 2) on these predictions (using hold-out targets as labels).
5. At prediction time: base classifiers predict → their outputs are fed to the meta-learner → final prediction.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC

stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5,    # use cross-validation to generate training data for meta-learner
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
```

### Pros

- Can squeeze out additional performance beyond voting classifiers.
- The meta-learner can discover optimal ways to weight and combine base predictions.
- Supports multi-layer stacking for even more complex ensembles.

### Cons

- More complex to implement and tune.
- Higher computational cost (training base classifiers + meta-learner).
- Increased risk of overfitting, especially with small datasets.
- Harder to interpret.

### Common Pitfalls

- Training the meta-learner on the same data used to train base classifiers — causes severe overfitting. Always use a hold-out set or cross-validation.
- Using an overly complex meta-learner (e.g., deep neural network). A simple model like Logistic Regression is usually sufficient and less prone to overfitting.
- Not diversifying the base classifiers — stacking benefits most from diverse, uncorrelated learners.
- Stacking too many layers without enough data to support them.

---

## Quick Comparison Table

| Method | Training | Parallelizable | Variance Reduction | Bias | Overfitting Risk |
|--------|----------|:-:|:-:|:-:|:-:|
| Voting | Independent | Yes | Moderate | Unchanged | Low |
| Bagging | Independent (bootstrap) | Yes | High | Slightly higher | Low |
| Pasting | Independent (no replacement) | Yes | Moderate | Unchanged | Low–Moderate |
| Random Forest | Independent (bagging + feature sampling) | Yes | High | Slightly higher | Low |
| Extra-Trees | Independent (random thresholds) | Yes | Very High | Slightly higher | Low |
| AdaBoost | Sequential | No | Moderate | Low | Moderate–High |
| Gradient Boosting | Sequential | No | High | Low | Moderate–High |
| Stacking | Layer-wise | Partially | High | Low | Moderate–High |
