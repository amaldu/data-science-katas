# Dimensionality Reduction — Q&A

---

## 1. What are the main motivations for reducing a dataset's dimensionality? What are the main drawbacks?

**Main motivations:**

- **Speed up training.** Fewer features means faster computation for most ML algorithms. Training time often scales with the number of features (sometimes quadratically or worse).
- **Combat the curse of dimensionality.** In high-dimensional spaces, data becomes sparse, distances lose meaning, and models are prone to overfitting. Reducing dimensions brings the data back into a regime where algorithms work well.
- **Data visualization.** Reducing to 2 or 3 dimensions allows you to plot and visually inspect the data, revealing clusters, outliers, and patterns that are invisible in high dimensions.
- **Reduce storage and memory.** Fewer dimensions means smaller datasets, which reduces storage costs and allows larger datasets to fit in memory.
- **Remove noise.** Many real-world features are redundant or noisy. Dimensionality reduction can strip away noise and highlight the underlying signal.
- **Reduce multicollinearity.** Highly correlated features can destabilize some models (e.g., linear regression). PCA produces uncorrelated components.

**Main drawbacks:**

- **Information loss.** Any dimensionality reduction discards some information. Even if 95% of variance is retained, the lost 5% could contain signal that matters for the task.
- **Reduced interpretability.** The new features (e.g., principal components) are linear combinations of the original features and often have no intuitive meaning. This makes model explanations harder.
- **Added complexity in the pipeline.** Dimensionality reduction adds a preprocessing step that must be tuned (number of components, kernel choice, etc.) and can introduce bugs or data leakage if done incorrectly.
- **Computational cost of the reduction itself.** Some methods (Kernel PCA, t-SNE, LLE) are computationally expensive, potentially negating the speed benefits.
- **May hurt performance.** If important features are discarded, the downstream model can perform worse than it would on the original data.

---

## 2. What is the curse of dimensionality?

The curse of dimensionality refers to the collection of problems that arise when working with data in high-dimensional spaces. As the number of features grows, several things break down:

1. **Sparsity:** The volume of the feature space grows exponentially with the number of dimensions. A fixed number of data points becomes increasingly sparse — there is too much "empty space" and too few data points to cover it. For example, 100 data points might densely cover a 1D line, but they would be hopelessly sparse in a 1000-dimensional space.

2. **Distance concentration:** In high dimensions, the ratio of the distance to the nearest neighbor vs. the farthest neighbor approaches 1. This means that all points become approximately equidistant from each other, making distance-based algorithms (KNN, k-means, DBSCAN) essentially useless.

3. **Exponential data requirement:** To maintain the same sampling density (and thus statistical reliability), the number of training instances needs to grow exponentially with the number of dimensions. This is almost never practical.

4. **Overfitting:** With many features and relatively few samples, models can easily find spurious patterns in the training data that don't generalize. The model memorizes noise instead of learning signal.

**Practical consequence:** A randomly sampled point in a high-dimensional unit hypercube is almost certainly near the boundary (not the center), and two random points are almost certainly far apart from each other. This breaks many of the geometric assumptions underlying ML algorithms.

---

## 3. Once a dataset's dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?

**It depends on the method, but in general, perfect reversal is NOT possible.**

**For PCA:**
You can approximately reverse the projection using the `inverse_transform` method. Since PCA projects data onto a lower-dimensional subspace, reversing means projecting back from the reduced space to the original space:

$$\mathbf{X}_{\text{reconstructed}} = \mathbf{X}_{\text{reduced}} \cdot \mathbf{W}_d^T$$

However, this reconstruction is **lossy** — the information discarded when dropping the lower-variance components is permanently lost. The reconstruction error equals the variance in the dropped components. With a 95% explained variance ratio, about 5% of the variance is irreversibly lost.

```python
pca = PCA(n_components=154)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)  # approximate, not exact
```

**For Kernel PCA:**
`inverse_transform` is available but requires `fit_inverse_transform=True`. It trains a separate model (kernel ridge regression) to learn the mapping from the reduced space back to the original space, so the reconstruction is even more approximate than regular PCA.

**For manifold learning methods (t-SNE, LLE, MDS):**
There is generally **no inverse transform available**. These methods find a low-dimensional embedding by optimizing a nonlinear objective, and the mapping from the reduced space back to the original space is not well-defined. The transformation is a one-way process.

**Bottom line:** Dimensionality reduction is inherently lossy. You can get an approximation of the original data back (for some methods), but you can never perfectly recover the original data once dimensions have been discarded.

---

## 4. Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?

**Yes, but it will likely produce poor results.**

Standard PCA is a **linear** method — it projects data onto a flat hyperplane. If the data lies on a nonlinear manifold (e.g., a Swiss roll, concentric circles), PCA will project the data onto the directions of maximum variance, which may:

- Squash distinct parts of the manifold on top of each other
- Fail to capture the actual low-dimensional structure
- Mix together data points that should be far apart on the manifold

**Example:** The Swiss roll is a 2D manifold embedded in 3D. PCA would project it onto a 2D plane, flattening the roll and overlapping points from different layers. The manifold structure is completely destroyed.

**Better alternatives for nonlinear data:**

| Method | Why |
|:---|:---|
| **Kernel PCA** | Applies the kernel trick to perform PCA in a higher-dimensional space, capturing nonlinear structure |
| **LLE** | Preserves local linear relationships, can unroll manifolds |
| **t-SNE** | Preserves local structure for visualization |
| **UMAP** | Preserves local + some global structure, faster than t-SNE |
| **Isomap** | Preserves geodesic distances along the manifold |

**However,** PCA can still be useful as a **first step** even with nonlinear data — for example, reducing from 10,000 dimensions to 50 dimensions with PCA, then applying a nonlinear method on the 50-dimensional data. This two-stage approach is common and can save significant computation time.

---

## 5. Suppose you perform PCA on a 1,000-dimensional dataset, setting the explained variance ratio to 95%. How many dimensions will the resulting dataset have?

**It depends entirely on the dataset.** There is no way to determine the answer without actually running PCA on the specific data.

The number of resulting dimensions depends on how the variance is distributed across the principal components:

- **If variance is highly concentrated** (e.g., the features are highly correlated), a few components will capture 95% of the variance. You might end up with, say, 10–50 dimensions.
- **If variance is spread evenly** across all 1,000 features (features are independent and equally variable), you would need close to 950 components to reach 95%. In the extreme case where all features contribute equally, you'd need exactly 950 dimensions.
- **In practice,** real-world datasets usually fall somewhere in between. For a typical dataset, you might expect anywhere from a few dozen to a few hundred dimensions.

**Example with code:**
```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
print(f"Reduced to {pca.n_components_} dimensions (from 1000)")
```

**Key insight:** The more correlated the features are, the fewer principal components are needed to capture 95% of the variance. The more independent they are, the more components are needed. This is why PCA works best when features are highly correlated — there is true redundancy to exploit.

---

## 6. In what cases would you use vanilla PCA, Incremental PCA, Randomized PCA, or Kernel PCA?

| Variant | When to Use | When NOT to Use |
|:---|:---|:---|
| **Vanilla PCA** | Small-to-medium datasets that fit in memory; you need the exact solution; the number of components is close to the number of features | Very large datasets that don't fit in memory; very high-dimensional data where $d \ll n$ |
| **Incremental PCA** | Large datasets that **don't fit in memory**; data arrives in a **stream** and you need online learning; you want to apply PCA to data stored on disk (using `np.memmap`) | Datasets that fit in memory (vanilla PCA is faster); when you need exact results (IPCA is approximate) |
| **Randomized PCA** | High-dimensional data where you want $d \ll n$ (large dimensionality reduction); you want the **fastest** computation and can tolerate very slight approximation | When $d$ is close to $n$ (not enough dimensionality reduction); when you need exact results |
| **Kernel PCA** | Data has **nonlinear structure** that linear PCA cannot capture; the kernel trick is appropriate (e.g., RBF for smooth manifolds, polynomial for polynomial relationships) | Linear data (use regular PCA — faster and simpler); very large datasets (kernel matrix is $O(m^2)$ memory); when interpretability is important |

**Decision flowchart:**

1. Is the data nonlinear? → **Kernel PCA** (or manifold learning)
2. Does the data fit in memory?
   - No → **Incremental PCA**
   - Yes → Is $d \ll n$? → **Randomized PCA** (Scikit-Learn uses this automatically)
   - Yes, and $d \approx n$ → **Vanilla PCA** (full SVD)

**Note:** Scikit-Learn's `PCA` with `svd_solver='auto'` automatically selects between full SVD and randomized SVD based on the data dimensions. In most practical cases, you can just use `PCA` and let Scikit-Learn choose.

---

## 7. How can you evaluate the performance of a dimensionality reduction algorithm on your dataset?

There is no single "accuracy" metric for dimensionality reduction. The best evaluation depends on what you're using dimensionality reduction for:

### Strategy 1: Downstream task performance (most common and reliable)

Use the reduced data as input to a downstream ML model and measure how that model performs. Compare with the model trained on the original data.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# With PCA
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', RandomForestClassifier()),
])
score_pca = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()

# Without PCA
pipe_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier()),
])
score_no_pca = cross_val_score(pipe_no_pca, X_train, y_train, cv=5, scoring='accuracy').mean()
```

If accuracy (or F1, AUC, etc.) is similar or better with fewer dimensions, the dimensionality reduction is working well.

### Strategy 2: Reconstruction error (for projection-based methods)

Measure how well the original data can be reconstructed from the reduced representation:

$$\text{Reconstruction Error} = \frac{1}{m} \sum_{i=1}^{m} \| \mathbf{x}_i - \hat{\mathbf{x}}_i \|^2$$

Lower reconstruction error means less information was lost. For PCA, this is directly related to the explained variance ratio.

### Strategy 3: Explained variance ratio (PCA only)

Check how much of the total variance is captured by the retained components. 95% is a common threshold, but the right value depends on the task.

### Strategy 4: Visual inspection (for visualization methods)

For methods like t-SNE and UMAP, visually inspect the 2D/3D embedding. Well-separated clusters with consistent colors (labels) indicate good preservation of structure.

### Strategy 5: Trustworthiness and continuity metrics

Quantitative metrics from `sklearn.manifold.trustworthiness` measure how well local neighborhoods are preserved:

```python
from sklearn.manifold import trustworthiness

score = trustworthiness(X_original, X_reduced, n_neighbors=5)
# Range [0, 1], higher is better
```

**Best practice:** Evaluate using the downstream task. The "best" dimensionality reduction is the one that helps your actual problem the most (or hurts it the least).

---

## 8. Does it make any sense to chain two different dimensionality reduction algorithms?

**Yes, absolutely.** Chaining dimensionality reduction algorithms is a common and useful practice. The most typical pattern is:

### PCA first, then a nonlinear method

Many nonlinear methods (t-SNE, LLE) are computationally expensive — $O(m^2)$ or $O(m^3)$. Running them on high-dimensional data is slow or infeasible. The standard approach is:

1. **PCA** to reduce from the original high-dimensional space to an intermediate dimensionality (~50–100 dimensions). This is fast and removes most of the redundancy.
2. **t-SNE / UMAP / LLE** on the PCA-reduced data to further reduce to 2–3 dimensions for visualization or to capture nonlinear structure.

```python
# Common pattern: PCA → t-SNE
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)
```

**Why this works:**
- PCA quickly removes the "easy" redundancy (correlated features) at low computational cost.
- The nonlinear method then works on a much smaller input, making it faster and often more effective (less noise to deal with).
- The PCA step doesn't lose much information (it captures 95%+ of the variance), so the nonlinear method sees essentially the same data.

**Other valid chains:**
- **PCA → Kernel PCA:** Use PCA to reduce to a manageable size, then Kernel PCA for nonlinear structure.
- **PCA → LLE:** Same idea — reduce first, then unroll the manifold.
- **Two PCA steps:** Less common but valid — e.g., PCA on different subsets of features, then a final PCA on the concatenated results.

**When it does NOT make sense:**
- Chaining two linear methods (e.g., PCA → PCA) is redundant — the second PCA just selects a subset of the already-optimal components. A single PCA with fewer components gives the same result.
- Chaining methods when the first step already achieves the target dimensionality.

---

## 9. PCA + Random Forest on MNIST — Speed and Accuracy Comparison

```python
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Split: first 60,000 train, remaining 10,000 test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# --- Train on original data ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

start = time.time()
rf.fit(X_train, y_train)
train_time_original = time.time() - start

accuracy_original = accuracy_score(y_test, rf.predict(X_test))
print(f"Original data (784 dimensions):")
print(f"  Training time: {train_time_original:.2f}s")
print(f"  Test accuracy: {accuracy_original:.4f}")

# --- Reduce with PCA (95% variance), then train ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"\nPCA reduced to {pca.n_components_} dimensions")

rf_pca = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

start = time.time()
rf_pca.fit(X_train_pca, y_train)
train_time_pca = time.time() - start

accuracy_pca = accuracy_score(y_test, rf_pca.predict(X_test_pca))
print(f"PCA-reduced data ({pca.n_components_} dimensions):")
print(f"  Training time: {train_time_pca:.2f}s")
print(f"  Test accuracy: {accuracy_pca:.4f}")

print(f"\nSpeedup: {train_time_original / train_time_pca:.1f}x faster")
print(f"Accuracy change: {accuracy_pca - accuracy_original:+.4f}")
```

**Expected results (approximate):**

| Metric | Original (784D) | PCA (~154D) |
|:---|:---|:---|
| Training time | ~30–60s | ~5–15s |
| Test accuracy | ~96.5–97% | ~94–95% |
| Dimensions | 784 | ~150 |

**Key observations:**
- **Training is significantly faster** with PCA (roughly 3–6x speedup, depending on hardware), because the Random Forest has fewer features to consider at each split.
- **Accuracy drops slightly** (typically 1–2 percentage points). This is because PCA discards some variance that contains discriminative information. The 5% "lost" variance includes some signal, not just noise.
- **The trade-off is often worthwhile** — much faster training for a small accuracy cost. In some cases, the accuracy drop is negligible.
- **Random Forests are relatively robust** to high dimensions, so the accuracy drop from PCA may be larger than for models that struggle more with high-dimensional data (e.g., KNN, logistic regression — where PCA can sometimes even improve accuracy).

---

## 10. t-SNE Visualization of MNIST + Comparison with Other Methods

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler

# Load MNIST (use a subset for speed)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Sample 10,000 instances for tractable computation
np.random.seed(42)
sample_idx = np.random.choice(len(X), 10000, replace=False)
X_sample = X[sample_idx]
y_sample = y[sample_idx]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Step 1: PCA to 50 dimensions (speeds up subsequent methods)
pca_50 = PCA(n_components=50, random_state=42)
X_pca50 = pca_50.fit_transform(X_scaled)

# --- t-SNE ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca50)

# --- PCA (2D) ---
pca_2d = PCA(n_components=2, random_state=42)
X_pca2d = pca_2d.fit_transform(X_scaled)

# --- LLE ---
# (use a smaller sample — LLE is slow)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X_pca50[:5000])
y_lle = y_sample[:5000]

# --- Plotting ---
def plot_embedding(X_2d, y_labels, title, ax):
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for digit in range(10):
        mask = y_labels == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[digit]], label=str(digit),
                   alpha=0.5, s=5)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

plot_embedding(X_tsne, y_sample, "t-SNE", axes[0])
plot_embedding(X_pca2d, y_sample, "PCA", axes[1])
plot_embedding(X_lle, y_lle, "LLE (5K samples)", axes[2])

axes[0].legend(markerscale=3, fontsize=8, loc='best')
plt.tight_layout()
plt.savefig("mnist_dimensionality_reduction.png", dpi=150, bbox_inches='tight')
plt.show()
```

**Expected observations:**

| Method | Visual Quality | What You See |
|:---|:---|:---|
| **t-SNE** | Excellent | Well-separated, tight clusters for each digit (0–9). Some digits (4 and 9, 3 and 5) may be close but still distinguishable. This is the gold standard for visualization. |
| **PCA** | Poor | Digits heavily overlap. PCA preserves global variance, not local cluster structure. You can see some separation (e.g., 0 and 1 are somewhat distinct) but most digits are mixed together. |
| **LLE** | Moderate | Some cluster structure is visible, but typically noisier than t-SNE. Results are sensitive to `n_neighbors`. |
| **MDS** | Moderate | Similar to PCA — preserves global distances, so clusters overlap significantly. Very slow to compute. |
| **UMAP** (if used) | Excellent | Similar quality to t-SNE, but preserves more global structure. Digit clusters are well-separated and the overall layout reflects digit similarity (e.g., 4 and 9 are near each other). |

**Key takeaways:**
- **t-SNE and UMAP** are far superior for visualization — they produce clear, well-separated clusters.
- **PCA** preserves global variance but fails to separate classes in 2D because the decision boundaries between digits are nonlinear.
- **LLE** can capture some manifold structure but is sensitive to hyperparameters and noise.
- **MDS** is too slow for datasets of this size and produces results similar to PCA.
- **Best practice for MNIST visualization:** PCA to ~50D, then t-SNE or UMAP to 2D. The PCA step removes noise and speeds up the nonlinear method without losing meaningful information.
