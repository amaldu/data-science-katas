# Classification Evaluation Metrics

Classification metrics evaluate how well a model assigns data points to the correct class. Unlike regression metrics (which measure continuous error), classification metrics deal with discrete outcomes — correct or incorrect predictions — and the different types of errors a model can make.

Where: $m$ = number of samples, $K$ = number of classes.

---

## The Confusion Matrix

The confusion matrix is the **foundation** of all classification metrics. It counts how many predictions fell into each category of correct/incorrect for each class.

**Binary classification (2 classes):**

```
                    Predicted
                  Positive    Negative
Actual Positive |    TP     |    FN    |
Actual Negative |    FP     |    TN    |
```

| Term | Meaning | Intuition |
|:---|:---|:---|
| **TP** (True Positive) | Predicted positive, actually positive | Correct alarm |
| **TN** (True Negative) | Predicted negative, actually negative | Correct silence |
| **FP** (False Positive) | Predicted positive, actually negative | False alarm (Type I error) |
| **FN** (False Negative) | Predicted negative, actually positive | Missed case (Type II error) |

**Python:**
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
print(cm)

# Visual display
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
```

---

## 1. Accuracy

**Definition:** The proportion of all predictions that are correct.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct predictions}}{\text{Total predictions}}$$

**Interpretation:**
- Range: 0 to 1 (or 0% to 100%)
- Accuracy = 0.95 means 95% of predictions are correct

**When to use:**
- Classes are **balanced** (roughly equal number of samples per class)
- All types of errors are equally costly

**When NOT to use:**
- **Imbalanced classes.** This is the most common pitfall. Example: in fraud detection where 99% of transactions are legitimate, a model that always predicts "not fraud" achieves 99% accuracy but catches zero fraud cases. Accuracy is misleading here.

**Advantages:**
- ✅ Simple and intuitive
- ✅ Single number summary
- ✅ Works well for balanced datasets

**Disadvantages:**
- ❌ **Misleading with imbalanced classes** — the dominant failure mode
- ❌ Treats all errors equally (a false positive counts the same as a false negative)
- ❌ Doesn't tell you which classes the model struggles with

**Python:**
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)

# From scratch
accuracy = np.mean(y_true == y_pred)
```

---

## 2. Precision

**Definition:** Of all the instances the model **predicted as positive**, what fraction are actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Intuition:** "When the model says YES, how often is it right?"

**Interpretation:**
- Precision = 0.90 means 90% of predicted positives are truly positive, and 10% are false alarms
- High precision = few false positives

**When to use:**
- When **false positives are costly:**
  - Spam detection: marking a legitimate email as spam means the user misses it
  - Criminal justice: convicting an innocent person
  - Medical treatment: prescribing a drug to someone who doesn't need it (side effects)

**Advantages:**
- ✅ Directly measures the reliability of positive predictions
- ✅ Critical when the cost of false positives is high

**Disadvantages:**
- ❌ Ignores false negatives entirely — a model that predicts positive only once (and gets it right) has perfect precision but misses all other positives
- ❌ Can be gamed by being overly conservative (predict positive rarely)

**Python:**
```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)

# From scratch
tp = np.sum((y_pred == 1) & (y_true == 1))
fp = np.sum((y_pred == 1) & (y_true == 0))
precision = tp / (tp + fp)
```

---

## 3. Recall (Sensitivity / True Positive Rate)

**Definition:** Of all the instances that are **actually positive**, what fraction did the model correctly identify?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Intuition:** "Of all the actual positives, how many did the model find?"

**Interpretation:**
- Recall = 0.95 means the model catches 95% of all positive cases, missing only 5%
- High recall = few false negatives

**When to use:**
- When **false negatives are costly:**
  - Disease detection: missing a cancer diagnosis could be fatal
  - Fraud detection: missing a fraudulent transaction costs money
  - Safety systems: failing to detect a defect in manufacturing

**Advantages:**
- ✅ Measures the model's ability to find all positive cases
- ✅ Critical when missing a positive case has severe consequences

**Disadvantages:**
- ❌ Ignores false positives — a model that predicts EVERYTHING as positive has perfect recall but terrible precision
- ❌ Can be gamed by being overly aggressive (predict positive always)

**Python:**
```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)

# From scratch
tp = np.sum((y_pred == 1) & (y_true == 1))
fn = np.sum((y_pred == 0) & (y_true == 1))
recall = tp / (tp + fn)
```

---

## 4. The Precision-Recall Trade-off

Precision and recall are **inversely related** — improving one typically hurts the other.

| Threshold | Precision | Recall | Effect |
|:---:|:---:|:---:|:---|
| **Low** (e.g., 0.3) | Low | High | Predicts many positives → catches most, but many false alarms |
| **Default** (0.5) | Balanced | Balanced | Standard operating point |
| **High** (e.g., 0.8) | High | Low | Predicts few positives → very reliable, but misses many |

**How it works:** Most classifiers output a probability. The **threshold** determines the cutoff:
- Probability $\geq$ threshold → predict positive
- Probability $<$ threshold → predict negative

Lowering the threshold catches more positives (higher recall) but also produces more false alarms (lower precision). Raising the threshold does the opposite.

**Python — Precision-Recall curve:**
```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

---

## 5. F1 Score

**Definition:** The **harmonic mean** of precision and recall. It balances both metrics into a single number, penalizing extreme imbalances.

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

**Why harmonic mean (not arithmetic)?** The harmonic mean penalizes extreme differences. If precision = 1.0 and recall = 0.0, the arithmetic mean would be 0.5 (seems OK), but the harmonic mean is 0.0 (correctly reflects that the model is useless).

**Interpretation:**
- Range: 0 to 1
- F1 = 1.0: perfect precision and recall
- F1 = 0.0: either precision or recall is zero
- High F1 requires both precision and recall to be high

**When to use:**
- **Imbalanced classes** where accuracy is misleading
- When you need a single metric that balances precision and recall
- When you can't decide between optimizing for precision or recall

**Variants:**
- **F0.5 Score:** Weighs precision more than recall (use when FP is more costly)
- **F2 Score:** Weighs recall more than precision (use when FN is more costly)

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

**Advantages:**
- ✅ Single metric that balances precision and recall
- ✅ Works well with imbalanced classes
- ✅ Penalizes models with extreme precision/recall imbalance

**Disadvantages:**
- ❌ Treats precision and recall as equally important (which may not match your problem)
- ❌ Ignores true negatives (TN doesn't appear in the formula)
- ❌ Threshold-dependent (changes with the classification threshold)

**Python:**
```python
from sklearn.metrics import f1_score, fbeta_score

f1 = f1_score(y_true, y_pred)
f05 = fbeta_score(y_true, y_pred, beta=0.5)  # precision-weighted
f2 = fbeta_score(y_true, y_pred, beta=2)      # recall-weighted
```

---

## 6. Specificity (True Negative Rate)

**Definition:** Of all the instances that are **actually negative**, what fraction did the model correctly identify as negative?

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Intuition:** "Of all the actual negatives, how many did the model correctly reject?"

**Interpretation:**
- Specificity = 0.98 means 98% of negative cases are correctly identified
- High specificity = few false positives (from the negative class perspective)

**When to use:**
- When correctly identifying negatives is important (e.g., medical screening — avoiding unnecessary invasive procedures on healthy patients)
- Used together with recall to plot the ROC curve

**Python:**
```python
# From scratch (no direct sklearn function)
tn = np.sum((y_pred == 0) & (y_true == 0))
fp = np.sum((y_pred == 1) & (y_true == 0))
specificity = tn / (tn + fp)
```

---

## 7. ROC Curve and AUC

### ROC Curve (Receiver Operating Characteristic)

**Definition:** A plot of **Recall (True Positive Rate)** vs. **False Positive Rate (1 - Specificity)** at all possible classification thresholds.

$$\text{TPR (Recall)} = \frac{TP}{TP + FN} \qquad \text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

The ROC curve shows the trade-off between catching positives (TPR) and generating false alarms (FPR) as you vary the threshold.

**How to read it:**
- **Top-left corner** (TPR=1, FPR=0) = perfect classifier
- **Diagonal line** (TPR = FPR) = random guessing
- **Below diagonal** = worse than random (model is inverted)

### AUC (Area Under the ROC Curve)

**Definition:** The area under the ROC curve. Summarizes the ROC curve into a single number.

**Interpretation:**
- AUC = 1.0: perfect classifier (separates all positives from negatives)
- AUC = 0.5: random guessing (no discriminative power)
- AUC < 0.5: worse than random (model predictions are inverted)
- AUC = 0.85: "there is an 85% chance that the model ranks a random positive instance higher than a random negative instance"

| AUC Range | Quality |
|:---:|:---|
| 0.90 – 1.00 | Excellent |
| 0.80 – 0.90 | Good |
| 0.70 – 0.80 | Fair |
| 0.60 – 0.70 | Poor |
| 0.50 – 0.60 | Fail (near random) |

**When to use:**
- When you want a **threshold-independent** evaluation (doesn't depend on a specific cutoff)
- Comparing multiple models on the same dataset
- When the operating threshold will be chosen later

**Advantages:**
- ✅ Threshold-independent — evaluates the model across all thresholds
- ✅ Works well even with moderate class imbalance
- ✅ Single number summary of discriminative ability
- ✅ Widely used and understood

**Disadvantages:**
- ❌ Can be **misleading with severe class imbalance** (use Precision-Recall AUC instead)
- ❌ Doesn't tell you the best threshold to use
- ❌ Treats all thresholds equally, even unrealistic ones

**Python:**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# AUC score (needs probabilities, not hard predictions)
auc = roc_auc_score(y_true, y_scores)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 8. Precision-Recall AUC (PR-AUC / Average Precision)

**Definition:** The area under the Precision-Recall curve. More informative than ROC-AUC for **imbalanced datasets**.

**When to use instead of ROC-AUC:**
- When the **positive class is rare** (e.g., 1% fraud, 0.1% disease)
- ROC-AUC can look optimistically high on imbalanced data because TNs dominate the FPR calculation. PR-AUC focuses entirely on the positive class.

**Interpretation:**
- PR-AUC = 1.0: perfect precision and recall at all thresholds
- Baseline for random classifier = proportion of positive class (not 0.5 like ROC-AUC)

**Python:**
```python
from sklearn.metrics import average_precision_score, precision_recall_curve

pr_auc = average_precision_score(y_true, y_scores)

precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
plt.plot(recalls, precisions, label=f'PR-AUC = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

---

## 9. Log Loss (Binary Cross-Entropy)

**Definition:** Measures how well the predicted **probabilities** match the true labels. Unlike accuracy/precision/recall which evaluate hard predictions (0 or 1), log loss evaluates the quality of the probability estimates.

$$\text{Log Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**Interpretation:**
- Lower = better. Log Loss = 0 means perfect probability estimates.
- Heavily penalizes **confident wrong predictions** (e.g., predicting 0.99 when the true label is 0).
- A model that outputs 0.5 for everything achieves Log Loss = $\ln(2) \approx 0.693$.

**When to use:**
- When the **calibration of probabilities** matters (not just the ranking)
- When you care about how confident the model is in its predictions
- As the training loss function for logistic regression and neural networks

**Advantages:**
- ✅ Evaluates probability quality, not just class predictions
- ✅ Heavily punishes overconfident wrong predictions
- ✅ Smooth and differentiable (used directly as a loss function)

**Disadvantages:**
- ❌ Not intuitive to interpret (what does Log Loss = 0.35 mean?)
- ❌ Sensitive to probability calibration — an uncalibrated model can have poor log loss even with good accuracy

**Python:**
```python
from sklearn.metrics import log_loss

ll = log_loss(y_true, y_pred_proba)
```

---

## 10. Cohen's Kappa

**Definition:** Measures how much better the classifier is compared to a classifier that guesses randomly based on class frequencies. It accounts for **agreement by chance**.

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where $p_o$ = observed accuracy, $p_e$ = expected accuracy from random guessing (based on class proportions).

**Interpretation:**
- $\kappa$ = 1.0: perfect agreement
- $\kappa$ = 0.0: no better than random (accounting for class distribution)
- $\kappa$ < 0.0: worse than random

| $\kappa$ Range | Agreement Level |
|:---:|:---|
| 0.81 – 1.00 | Almost perfect |
| 0.61 – 0.80 | Substantial |
| 0.41 – 0.60 | Moderate |
| 0.21 – 0.40 | Fair |
| 0.00 – 0.20 | Slight |

**When to use:**
- When classes are **imbalanced** and you want a metric that accounts for chance agreement
- Comparing annotator/model agreement
- When accuracy is misleading due to class distribution

**Python:**
```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
```

---

## Multiclass Extensions

Most binary metrics can be extended to multiclass problems using three averaging strategies:

| Strategy | How It Works | When to Use |
|:---|:---|:---|
| **Macro** | Compute metric per class, then average (equal weight per class) | All classes equally important |
| **Weighted** | Compute metric per class, average weighted by class support (sample count) | Account for class imbalance |
| **Micro** | Aggregate TP, FP, FN across all classes, then compute metric globally | Large datasets, overall performance |

```python
from sklearn.metrics import f1_score, precision_score, recall_score

# Multiclass
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
f1_micro = f1_score(y_true, y_pred, average='micro')

# Per-class breakdown
f1_per_class = f1_score(y_true, y_pred, average=None)
```

**Complete classification report:**
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
```

---

## How to Choose the Right Metric

| Scenario | Recommended Metric | Why |
|:---|:---|:---|
| **Balanced classes**, simple evaluation | **Accuracy** | All errors are equal, classes are balanced |
| **Imbalanced classes** | **F1 Score**, **PR-AUC** | Accuracy is misleading |
| **False positives are costly** (spam, innocent conviction) | **Precision** | Minimize false alarms |
| **False negatives are costly** (disease, fraud) | **Recall** | Don't miss positive cases |
| **Need a single balanced metric** | **F1 Score** | Balances precision and recall |
| **Comparing models** (threshold-independent) | **ROC-AUC** or **PR-AUC** | Evaluates across all thresholds |
| **Imbalanced + threshold-independent** | **PR-AUC** | Better than ROC-AUC for rare positives |
| **Probability calibration matters** | **Log Loss** | Evaluates probability quality |
| **Chance-corrected evaluation** | **Cohen's Kappa** | Accounts for random agreement |
| **Multiclass with equal class importance** | **Macro-F1** | Treats all classes equally |
| **Multiclass with proportional importance** | **Weighted-F1** | Accounts for class sizes |

**Best practice:** Report multiple metrics:
- **Confusion matrix** (full picture) + **F1** (balanced summary) + **ROC-AUC** (threshold-independent)

---

## Evaluation Metrics Summary

| Metric | Formula | Range | Threshold-dependent? | Best For |
|:---|:---|:---|:---|:---|
| **Accuracy** | $(TP+TN) / (TP+TN+FP+FN)$ | [0, 1] | Yes | Balanced classes |
| **Precision** | $TP / (TP+FP)$ | [0, 1] | Yes | Minimizing false positives |
| **Recall** | $TP / (TP+FN)$ | [0, 1] | Yes | Minimizing false negatives |
| **F1 Score** | $2 \cdot P \cdot R / (P+R)$ | [0, 1] | Yes | Imbalanced classes |
| **Specificity** | $TN / (TN+FP)$ | [0, 1] | Yes | Identifying negatives |
| **ROC-AUC** | Area under ROC curve | [0, 1] | No | Model comparison |
| **PR-AUC** | Area under PR curve | [0, 1] | No | Imbalanced + comparison |
| **Log Loss** | $-\frac{1}{m}\sum[y\log\hat{p}+(1-y)\log(1-\hat{p})]$ | [0, ∞) | No | Probability calibration |
| **Cohen's Kappa** | $(p_o - p_e)/(1 - p_e)$ | [-1, 1] | Yes | Chance-corrected accuracy |

**Time Complexity:** All classification metrics are $O(m)$ — a single pass over the predictions. Computing ROC/PR curves requires sorting predictions: $O(m \log m)$.

---

## Python Quick Reference

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

# Hard predictions (0 or 1)
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1:        {f1_score(y_true, y_pred):.4f}")
print(f"Kappa:     {cohen_kappa_score(y_true, y_pred):.4f}")

# Probability predictions
print(f"ROC-AUC:   {roc_auc_score(y_true, y_scores):.4f}")
print(f"PR-AUC:    {average_precision_score(y_true, y_scores):.4f}")
print(f"Log Loss:  {log_loss(y_true, y_pred_proba):.4f}")

# Full report
print(classification_report(y_true, y_pred))
```
