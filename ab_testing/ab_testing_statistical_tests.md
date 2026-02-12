# Statistical Tests for A/B Testing

> A comprehensive guide to every statistical test commonly used in A/B testing — what it is, how it works, when to use it, and how to implement it in Python.

Where: $n$ = sample size, $k$ = number of groups, $\alpha$ = significance level (typically 0.05).

---

## Quick Decision Guide

```
What type of metric are you testing?
│
├── BINARY / PROPORTION (conversion rate, CTR, signup rate)
│   ├── 2 groups, large n (>30 per group) → Z-test for Proportions
│   ├── 2 groups, small n (<30 or expected freq <5) → Fisher's Exact Test
│   ├── 2+ groups, large n → Chi-Square Test
│   └── Any size, want exact results → Permutation Test
│
├── CONTINUOUS (revenue, time on page, order value)
│   ├── 2 groups, normal data, equal variance → Two-Sample t-test
│   ├── 2 groups, normal data, unequal variance → Welch's t-test (DEFAULT)
│   ├── 2 groups, non-normal or skewed → Mann-Whitney U Test
│   ├── 2 groups, non-normal, want CI → Bootstrap Test
│   ├── 3+ groups, normal data → ANOVA (or Welch's ANOVA)
│   └── 3+ groups, non-normal data → Kruskal-Wallis Test
│
├── ORDINAL (ratings 1-5, satisfaction scales)
│   ├── 2 groups → Mann-Whitney U Test
│   └── 3+ groups → Kruskal-Wallis Test
│
└── UNSURE about assumptions → Permutation Test or Bootstrap Test
```

---

## Parametric Tests

Parametric tests assume the data follows a specific distribution (usually normal). They are more powerful (better at detecting true effects) when their assumptions hold.

---

### 1. Z-test for Proportions

**What it is:** A hypothesis test that compares two proportions (e.g., conversion rates) to determine if they are statistically different. It uses the normal approximation to the binomial distribution.

**How it works:**
1. Compute the conversion rate for each group: $\hat{p}_A = x_A / n_A$, $\hat{p}_B = x_B / n_B$
2. Compute the pooled proportion: $\hat{p} = (x_A + x_B) / (n_A + n_B)$
3. Calculate the standard error: $SE = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}$
4. Compute the Z-statistic: $Z = \frac{\hat{p}_B - \hat{p}_A}{SE}$
5. Compare Z to the critical value or compute the p-value from the standard normal distribution.

**Test statistic:**

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}$$

**Assumptions:**
- Data is binary (success/failure, 0/1)
- Observations are independent
- Sample size is large enough: $n\hat{p} \geq 5$ and $n(1-\hat{p}) \geq 5$ (normal approximation holds)

**When to use in A/B testing:**
- Comparing **conversion rates**, click-through rates (CTR), signup rates, bounce rates
- Both groups have at least 30 observations (ideally hundreds+)
- The most common test in A/B testing for proportion-based metrics

**Advantages:**
- ✅ Simple and fast to compute
- ✅ Well-understood, widely accepted in industry
- ✅ Works great for large samples (which most A/B tests have)
- ✅ Directly gives you a confidence interval for the difference in proportions

**Disadvantages:**
- ❌ Requires large $n$ — breaks down with small samples
- ❌ Assumes known (or well-estimated) population variance
- ❌ Normal approximation may be poor for very small or very large proportions (e.g., 0.01%)

**Python:**
```python
from statsmodels.stats.proportion import proportions_ztest

# x = number of successes, n = number of trials
x = [conversions_A, conversions_B]
n = [total_A, total_B]

z_stat, p_value = proportions_ztest(x, n, alternative='two-sided')
print(f"Z-statistic: {z_stat:.4f}, p-value: {p_value:.4f}")
```

---

### 2. Two-Sample t-test (Student's t-test)

**What it is:** A hypothesis test that compares the **means** of two independent groups to determine if they are statistically different. Uses the t-distribution, which accounts for the extra uncertainty of estimating variance from the sample.

**How it works:**
1. Compute the mean and variance for each group: $\bar{x}_A, s_A^2$ and $\bar{x}_B, s_B^2$
2. Compute the pooled standard error (assuming equal variances): $SE = s_p \sqrt{\frac{1}{n_A} + \frac{1}{n_B}}$
3. Compute the t-statistic: $t = \frac{\bar{x}_B - \bar{x}_A}{SE}$
4. Compare to the t-distribution with $n_A + n_B - 2$ degrees of freedom.

**Test statistic:**

$$t = \frac{\bar{x}_B - \bar{x}_A}{s_p \sqrt{\frac{1}{n_A} + \frac{1}{n_B}}}$$

Where $s_p = \sqrt{\frac{(n_A-1)s_A^2 + (n_B-1)s_B^2}{n_A + n_B - 2}}$ is the pooled standard deviation.

**Assumptions:**
- Data is continuous
- Observations are independent
- Both groups are approximately normally distributed (or $n > 30$ by CLT)
- **Equal variances** in both groups (homoscedasticity)

**When to use in A/B testing:**
- Comparing **average order value**, session duration, pages per visit
- When you have reason to believe variances are equal across groups
- Small to moderate sample sizes where the normal approximation (Z-test) is less reliable

**Advantages:**
- ✅ Works with smaller samples than the Z-test
- ✅ Doesn't require known population variance (estimates from data)
- ✅ More accurate than Z-test for small $n$

**Disadvantages:**
- ❌ Assumes **equal variances** — often violated in practice
- ❌ Assumes normality (though robust for $n > 30$ via CLT)
- ❌ Sensitive to outliers (means and variances are affected)
- ❌ In practice, **Welch's t-test is almost always preferred**

**Python:**
```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

---

### 3. Welch's t-test

**What it is:** A modified version of the two-sample t-test that does **not assume equal variances**. It adjusts the degrees of freedom using the Welch-Satterthwaite equation, making it robust to unequal group variances.

**How it works:**
Same as the two-sample t-test, but:
- Uses **unpooled** standard error: $SE = \sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}$
- Adjusts degrees of freedom with the Welch-Satterthwaite formula (usually non-integer)

**Test statistic:**

$$t = \frac{\bar{x}_B - \bar{x}_A}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}$$

**Degrees of freedom (Welch-Satterthwaite):**

$$df = \frac{\left(\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}\right)^2}{\frac{(s_A^2/n_A)^2}{n_A-1} + \frac{(s_B^2/n_B)^2}{n_B-1}}$$

**Assumptions:**
- Data is continuous
- Observations are independent
- Both groups are approximately normally distributed (or $n > 30$)
- **Does NOT require equal variances** — this is the key advantage

**When to use in A/B testing:**
- **The recommended default** for comparing continuous means in A/B tests
- Comparing revenue per user, time on site, number of actions
- Whenever you're unsure whether variances are equal (which is almost always)

**Advantages:**
- ✅ **No equal variance assumption** — robust to unequal group variances
- ✅ Performs almost as well as the equal-variance t-test when variances are actually equal
- ✅ Recommended as the default by most statisticians
- ✅ The loss of power compared to the equal-variance t-test is negligible

**Disadvantages:**
- ❌ Still assumes normality (though robust for large $n$)
- ❌ Still sensitive to outliers
- ❌ Slightly less powerful than equal-variance t-test when variances truly are equal (negligible in practice)

**Python:**
```python
from scipy import stats

# equal_var=False makes it Welch's t-test (THIS IS THE DEFAULT YOU SHOULD USE)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

> **Best practice:** Always use Welch's t-test (`equal_var=False`) as your default. There is almost never a good reason to use the equal-variance t-test in A/B testing.

---

### 4. One-Way ANOVA (Analysis of Variance)

**What it is:** A test that compares the **means of 3 or more groups** simultaneously. Instead of running multiple pairwise t-tests (which inflates the false positive rate), ANOVA tests whether at least one group mean differs from the others.

**How it works:**
1. Compute the overall mean $\bar{x}$ across all groups.
2. Compute **between-group variance** (how different are the group means from each other?).
3. Compute **within-group variance** (how spread out is the data within each group?).
4. Compute the F-statistic: $F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$
5. If $F$ is large, the group means are more different than expected by chance.

**Test statistic:**

$$F = \frac{MS_{between}}{MS_{within}} = \frac{SS_{between} / (k-1)}{SS_{within} / (N-k)}$$

Where $k$ = number of groups, $N$ = total observations.

**Assumptions:**
- Data is continuous
- Observations are independent
- Each group is approximately normally distributed
- **Equal variances** across groups (use Welch's ANOVA if violated)

**When to use in A/B testing:**
- **Multi-variant tests (A/B/C/D)** — comparing 3+ design variants
- Testing multiple price points, multiple headline variations
- When you want to control the overall Type I error rate

**Important:** ANOVA only tells you that **at least one group is different** — not which one. Use **post-hoc tests** (Tukey HSD, Bonferroni) to find which specific pairs differ.

**Advantages:**
- ✅ Tests multiple groups in one test (controls family-wise error rate)
- ✅ More powerful than running many pairwise t-tests
- ✅ Well-established with clear post-hoc procedures

**Disadvantages:**
- ❌ Assumes equal variances (use Welch's ANOVA if violated)
- ❌ Assumes normality
- ❌ Only detects that groups differ — doesn't tell you which ones (need post-hoc)
- ❌ Sensitive to outliers

**Welch's ANOVA:** Use when variances are unequal — does not assume homoscedasticity.

**Python:**
```python
from scipy import stats

# Standard ANOVA (assumes equal variances)
f_stat, p_value = stats.f_oneway(group_a, group_b, group_c)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

# If significant, run post-hoc Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

data = np.concatenate([group_a, group_b, group_c])
labels = ['A']*len(group_a) + ['B']*len(group_b) + ['C']*len(group_c)

tukey = pairwise_tukeyhsd(data, labels, alpha=0.05)
print(tukey)
```

---

### 5. Paired t-test

**What it is:** A test that compares two related measurements on the **same subjects** (before/after, same user exposed to both conditions). It tests whether the mean difference within pairs is zero.

**How it works:**
1. Compute the difference $d_i = x_{i,B} - x_{i,A}$ for each paired observation.
2. Test whether the mean of these differences is zero using a one-sample t-test on $d$.

**Test statistic:**

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

Where $\bar{d}$ = mean of differences, $s_d$ = standard deviation of differences, $n$ = number of pairs.

**Assumptions:**
- Paired (dependent) observations
- Differences are approximately normally distributed

**When to use in A/B testing:**
- **Within-subject designs** — same user sees both A and B (crossover design)
- Before/after comparisons on the same users
- Matched-pair experiments (each user in treatment is matched with a similar control user)

**Advantages:**
- ✅ More powerful than independent t-test because it controls for individual variability
- ✅ Requires fewer subjects to detect the same effect

**Disadvantages:**
- ❌ Requires paired data — not always available in A/B testing
- ❌ Carryover effects can bias results in crossover designs

**Python:**
```python
from scipy import stats

t_stat, p_value = stats.ttest_rel(before, after)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

---

### 6. Pearson's Correlation Test

**What it is:** Tests whether there is a statistically significant **linear relationship** between two continuous variables. Outputs a correlation coefficient $r \in [-1, 1]$ and a p-value.

**Test statistic:**

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

**When to use in A/B testing:**
- Checking if two metrics are correlated (e.g., time on site and conversion rate)
- Validating metric relationships in post-analysis
- Not a primary A/B test comparison, but useful for **metric exploration**

**Python:**
```python
from scipy import stats

r, p_value = stats.pearsonr(metric_a, metric_b)
print(f"Correlation: {r:.4f}, p-value: {p_value:.4f}")
```

---

## Non-Parametric Tests

Non-parametric tests make **fewer assumptions** about the data distribution. They are generally less powerful than parametric tests when parametric assumptions hold, but are more robust when they don't.

---

### 7. Chi-Square Test of Independence

**What it is:** Tests whether there is a significant association between two categorical variables. In A/B testing, it tests whether the distribution of outcomes (e.g., converted vs. not converted) differs across groups.

**How it works:**
1. Build a **contingency table** of observed frequencies.
2. Calculate **expected frequencies** under the null hypothesis (independence): $E_{ij} = \frac{R_i \times C_j}{N}$
3. Compute $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$
4. Compare to the chi-square distribution with $(r-1)(c-1)$ degrees of freedom.

**Test statistic:**

$$\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where $O_{ij}$ = observed count, $E_{ij}$ = expected count, $r$ = rows, $c$ = columns.

**Assumptions:**
- Observations are independent
- Expected frequency $\geq 5$ in each cell (if not, use Fisher's Exact Test)
- Data is categorical (or binned continuous)

**When to use in A/B testing:**
- Comparing **categorical outcomes** across groups (click vs. no-click, plan A vs. plan B vs. plan C)
- Testing whether user **preference distributions** differ between variants
- Multi-category comparisons (not just binary)

**Advantages:**
- ✅ Handles categorical data with **multiple categories** (not just 2)
- ✅ Easy to compute and interpret
- ✅ No normality assumption

**Disadvantages:**
- ❌ Requires expected frequency $\geq 5$ per cell
- ❌ **Doesn't show direction** of effect — only that an association exists
- ❌ Sensitive to sample size — very large samples may find trivially small effects significant
- ❌ Does not indicate which specific cells deviate (need post-hoc residual analysis)

**Python:**
```python
from scipy.stats import chi2_contingency
import numpy as np

# Contingency table: rows = groups, columns = outcomes
observed = np.array([
    [conversions_A, total_A - conversions_A],
    [conversions_B, total_B - conversions_B]
])

chi2, p_value, dof, expected = chi2_contingency(observed)
print(f"Chi-square: {chi2:.4f}, p-value: {p_value:.4f}, dof: {dof}")
```

---

### 8. Fisher's Exact Test

**What it is:** An **exact** test for association in a 2x2 contingency table. Unlike the chi-square test, it doesn't rely on a large-sample approximation — it calculates the exact probability of obtaining the observed (or more extreme) table under the null hypothesis using the hypergeometric distribution.

**How it works:**
1. Fix the row and column totals (marginals) of the 2x2 table.
2. Enumerate all possible tables with those marginals.
3. Calculate the probability of each table under the null hypothesis.
4. Sum probabilities of tables as extreme or more extreme than the observed table.

**Exact probability for a 2x2 table:**

$$p = \frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{N}{a+c}}$$

Where $a, b, c, d$ are the four cells of the 2x2 table and $N = a + b + c + d$.

**Assumptions:**
- 2x2 contingency table (2 groups x 2 outcomes)
- Fixed marginal totals
- Independent observations

**When to use in A/B testing:**
- **Small sample sizes** — pilot tests, early experiments
- When any expected cell count is $< 5$ (chi-square approximation fails)
- **Rare events** — very low conversion rates where counts are small
- When you need an exact p-value rather than an approximation

**Advantages:**
- ✅ **Exact** — no large-sample approximation needed
- ✅ Works with very small samples (even $n = 10$)
- ✅ No minimum expected frequency requirement
- ✅ Always valid, regardless of sample size

**Disadvantages:**
- ❌ **Computationally expensive** for large tables (though modern libraries handle this)
- ❌ Practically limited to **2x2 tables** (extensions exist but are rarely used)
- ❌ Conservative — may have less power than chi-square for large samples

**Python:**
```python
from scipy.stats import fisher_exact
import numpy as np

table = np.array([
    [conversions_A, total_A - conversions_A],
    [conversions_B, total_B - conversions_B]
])

odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
print(f"Odds ratio: {odds_ratio:.4f}, p-value: {p_value:.4f}")
```

---

### 9. Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

**What it is:** A non-parametric test that compares two independent groups by **ranking** all observations and testing whether one group tends to have higher ranks. It tests whether the distributions of the two groups are different (specifically, whether one group stochastically dominates the other).

**How it works:**
1. Combine all observations from both groups and **rank** them (1 = smallest).
2. Sum the ranks for each group: $R_A$ and $R_B$.
3. Compute the U statistic: $U = R_A - \frac{n_A(n_A+1)}{2}$
4. For large samples, $U$ is approximately normally distributed.

**Test statistic:**

$$U = R_A - \frac{n_A(n_A+1)}{2}$$

Where $R_A$ is the sum of ranks in group A. For large $n$, use the Z-approximation:

$$Z = \frac{U - \frac{n_A n_B}{2}}{\sqrt{\frac{n_A n_B (n_A + n_B + 1)}{12}}}$$

**Assumptions:**
- Observations are independent
- Data is at least ordinal (can be ranked)
- Both distributions have the same shape (tests for shift in location)

**When to use in A/B testing:**
- **Non-normal, skewed data** — revenue per user, time on site (often right-skewed)
- **Ordinal data** — user ratings (1-5 stars), satisfaction scores (1-10)
- When outliers are a concern (ranks are robust to extreme values)
- When sample sizes are too small for CLT to kick in

**Advantages:**
- ✅ **No normality assumption** — works with any distribution shape
- ✅ **Robust to outliers** — uses ranks, not raw values
- ✅ Works with ordinal data
- ✅ Valid for small samples

**Disadvantages:**
- ❌ **Less powerful** than the t-test when data is actually normal (wastes information by ranking)
- ❌ Tests whether distributions differ, not specifically whether **means** differ
- ❌ Does not directly provide a confidence interval for the mean difference
- ❌ Assumes same distribution shape — if shapes differ, interpretation is tricky

**Python:**
```python
from scipy import stats

u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
print(f"U-statistic: {u_stat:.4f}, p-value: {p_value:.4f}")
```

---

### 10. Kruskal-Wallis H Test

**What it is:** The non-parametric equivalent of one-way ANOVA. Tests whether **3 or more independent groups** come from the same distribution. Like ANOVA, it only tells you that at least one group is different — not which one.

**How it works:**
1. Rank all observations across all groups combined.
2. Compute the H statistic based on the deviation of group rank means from the overall rank mean.
3. Compare H to the chi-square distribution with $k-1$ degrees of freedom.

**Test statistic:**

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

Where $R_i$ = sum of ranks in group $i$, $n_i$ = size of group $i$, $N$ = total observations.

**Assumptions:**
- Observations are independent
- Data is at least ordinal
- Groups have similar distribution shapes

**When to use in A/B testing:**
- **Multi-variant tests (A/B/C/D)** with non-normal data
- Comparing 3+ groups on ordinal or skewed continuous metrics

**Post-hoc:** If significant, use **Dunn's test** to determine which specific pairs differ.

**Advantages:**
- ✅ Non-parametric — no normality assumption
- ✅ Handles 3+ groups
- ✅ Robust to outliers

**Disadvantages:**
- ❌ Less powerful than ANOVA when normality holds
- ❌ Only indicates groups differ, not which ones
- ❌ Requires post-hoc testing (Dunn's test with Bonferroni correction)

**Python:**
```python
from scipy import stats

h_stat, p_value = stats.kruskal(group_a, group_b, group_c)
print(f"H-statistic: {h_stat:.4f}, p-value: {p_value:.4f}")

# If significant, use Dunn's test for pairwise comparisons
# pip install scikit-posthocs
import scikit_posthocs as sp
import pandas as pd

data = pd.DataFrame({
    'value': list(group_a) + list(group_b) + list(group_c),
    'group': ['A']*len(group_a) + ['B']*len(group_b) + ['C']*len(group_c)
})

dunn_results = sp.posthoc_dunn(data, val_col='value', group_col='group', p_adjust='bonferroni')
print(dunn_results)
```

---

### 11. Wilcoxon Signed-Rank Test

**What it is:** The non-parametric equivalent of the **paired t-test**. Tests whether the median difference between paired observations is zero. Uses ranks of the absolute differences.

**How it works:**
1. Compute differences $d_i = x_{i,B} - x_{i,A}$ for each pair.
2. Remove zero differences.
3. Rank the absolute differences.
4. Sum the ranks of positive differences ($W^+$) and negative differences ($W^-$).
5. The test statistic is $W = \min(W^+, W^-)$.

**Assumptions:**
- Paired (dependent) observations
- Differences are symmetric around the median

**When to use in A/B testing:**
- **Within-subject / before-after designs** with non-normal data
- Matched-pair experiments where difference distributions are skewed
- Ordinal paired data

**Advantages:**
- ✅ No normality assumption on the differences
- ✅ Robust to outliers
- ✅ Works with ordinal paired data

**Disadvantages:**
- ❌ Less powerful than paired t-test when normality holds
- ❌ Assumes symmetric distribution of differences

**Python:**
```python
from scipy import stats

w_stat, p_value = stats.wilcoxon(before, after, alternative='two-sided')
print(f"W-statistic: {w_stat:.4f}, p-value: {p_value:.4f}")
```

---

## Resampling-Based Tests

These modern methods make **minimal assumptions** by using the data itself to build the reference distribution. They are increasingly popular in industry for A/B testing.

---

### 12. Permutation Test (Randomization Test)

**What it is:** A non-parametric, exact test that determines significance by **randomly re-assigning** observations to groups and computing the test statistic for each reassignment. The p-value is the proportion of permutations that produce a test statistic as extreme as the observed one.

**How it works:**
1. Compute the observed test statistic (e.g., difference in means) from the original data.
2. **Shuffle** the group labels randomly (keeping group sizes fixed).
3. Compute the test statistic for this permuted dataset.
4. Repeat steps 2-3 many times (e.g., 10,000 permutations).
5. The p-value = proportion of permuted statistics $\geq$ the observed statistic.

**The key idea:** Under the null hypothesis (no treatment effect), the group labels are arbitrary — any assignment would be equally likely. So we simulate what the test statistic distribution looks like if there's truly no effect.

**Assumptions:**
- Observations are **exchangeable** under the null hypothesis (the labels can be swapped without changing the joint distribution)
- Independent observations

**When to use in A/B testing:**
- When parametric assumptions (normality, equal variance) are questionable
- For **any type of metric** — proportions, means, medians, ratios
- When you want to test **custom metrics** that don't have a standard test (e.g., 90th percentile of revenue)
- When sample sizes are small and you want exact results
- As a **validation** of parametric test results

**Advantages:**
- ✅ **No distributional assumptions** — works for any data type
- ✅ Can test **any statistic** (mean, median, ratio, percentile — anything you can compute)
- ✅ Exact for small samples (enumerate all permutations) or approximate for large samples
- ✅ Intuitive and easy to explain
- ✅ Increasingly popular at top tech companies (Google, Microsoft, Netflix)

**Disadvantages:**
- ❌ **Computationally expensive** — requires thousands of permutations
- ❌ Does not directly produce a confidence interval (use bootstrap for that)
- ❌ Results vary slightly between runs (set a random seed for reproducibility)
- ❌ Assumes exchangeability, which may not hold in all designs

**Python:**
```python
import numpy as np

def permutation_test(group_a, group_b, n_permutations=10000, seed=42):
    """Two-sided permutation test for difference in means."""
    rng = np.random.default_rng(seed)
    
    observed_diff = np.mean(group_b) - np.mean(group_a)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[n_a:]) - np.mean(combined[:n_a])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    
    p_value = count_extreme / n_permutations
    return observed_diff, p_value

diff, p_val = permutation_test(group_a, group_b)
print(f"Observed difference: {diff:.4f}, p-value: {p_val:.4f}")
```

**Vectorized (faster) version:**
```python
import numpy as np

def permutation_test_vectorized(group_a, group_b, n_permutations=10000, seed=42):
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(group_b) - np.mean(group_a)
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    
    perm_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(combined)
        perm_diffs[i] = np.mean(combined[n_a:]) - np.mean(combined[:n_a])
    
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value
```

---

### 13. Bootstrap Test

**What it is:** A resampling method that estimates the **sampling distribution** of a statistic by repeatedly drawing samples **with replacement** from the observed data. Unlike the permutation test (which gives a p-value), the bootstrap's main strength is producing **confidence intervals** for any statistic.

**How it works:**
1. From group A (size $n_A$), draw $n_A$ observations with replacement → compute statistic.
2. From group B (size $n_B$), draw $n_B$ observations with replacement → compute statistic.
3. Compute the difference in statistics.
4. Repeat steps 1-3 many times (e.g., 10,000 iterations).
5. The distribution of differences gives you a confidence interval.
6. If the CI excludes 0, the difference is significant.

**Assumptions:**
- The sample is representative of the population
- Independent observations (within each group)

**When to use in A/B testing:**
- When you need a **confidence interval** for a non-standard metric
- For **highly skewed data** (revenue, purchase amounts)
- For **complex metrics** like ratios (revenue per session), percentiles, or custom KPIs
- When parametric formulas for standard errors don't exist or are unreliable

**Advantages:**
- ✅ **No distributional assumptions**
- ✅ Produces confidence intervals for **any statistic** (mean, median, ratio, percentile, etc.)
- ✅ Handles skewed data well
- ✅ Very flexible — works with arbitrarily complex metrics
- ✅ Widely used in industry (Google, Uber, Airbnb)

**Disadvantages:**
- ❌ Computationally expensive (thousands of resamples)
- ❌ Can be unreliable with very small samples
- ❌ Multiple CI methods exist (percentile, BCa, studentized) — choosing can be confusing
- ❌ Slightly biased for small samples (BCa correction helps)

**Python:**
```python
import numpy as np

def bootstrap_ci(group_a, group_b, stat_func=np.mean, n_bootstrap=10000, 
                 confidence=0.95, seed=42):
    """Bootstrap confidence interval for the difference in a statistic."""
    rng = np.random.default_rng(seed)
    
    observed_diff = stat_func(group_b) - stat_func(group_a)
    boot_diffs = np.empty(n_bootstrap)
    
    for i in range(n_bootstrap):
        boot_a = rng.choice(group_a, size=len(group_a), replace=True)
        boot_b = rng.choice(group_b, size=len(group_b), replace=True)
        boot_diffs[i] = stat_func(boot_b) - stat_func(boot_a)
    
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    p_value = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    
    return observed_diff, (ci_lower, ci_upper), p_value

# Example: Compare means
diff, ci, p = bootstrap_ci(group_a, group_b, stat_func=np.mean)
print(f"Difference: {diff:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], p-value: {p:.4f}")

# Example: Compare medians
diff, ci, p = bootstrap_ci(group_a, group_b, stat_func=np.median)
print(f"Median diff: {diff:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# Example: Compare 90th percentile
diff, ci, p = bootstrap_ci(group_a, group_b, 
                            stat_func=lambda x: np.percentile(x, 90))
```

---

## Bayesian Tests

---

### 14. Bayesian A/B Test (Beta-Binomial Model for Proportions)

**What it is:** Instead of producing a p-value, the Bayesian approach computes the **posterior probability** that one variant is better than another. For proportions, it uses the Beta-Binomial conjugate model to analytically compute the posterior distribution of each group's conversion rate.

**How it works:**
1. Start with a prior belief about the conversion rate: $p \sim \text{Beta}(\alpha_0, \beta_0)$
2. Observe data: $x$ conversions out of $n$ trials.
3. Update to posterior: $p \mid x \sim \text{Beta}(\alpha_0 + x, \beta_0 + n - x)$
4. Compare the two posterior distributions to get $P(\text{B is better than A})$.

**When to use in A/B testing:**
- You want a direct probability: "There is an 87% chance B is better than A"
- You need to make decisions before reaching classical sample sizes
- Business stakeholders prefer intuitive probability statements over p-values
- You want to incorporate prior knowledge from previous experiments

**Advantages:**
- ✅ Intuitive interpretation — direct probability of one variant being better
- ✅ Can incorporate prior knowledge from previous experiments
- ✅ No fixed sample size required — can make decisions at any time
- ✅ Natural framework for expected loss / risk analysis
- ✅ No multiple-testing penalty for peeking at results (with proper implementation)

**Disadvantages:**
- ❌ Requires choosing a prior (though weakly informative priors work well)
- ❌ Less familiar to some teams — harder to get buy-in
- ❌ For continuous metrics, conjugate models may not exist (need MCMC)
- ❌ "Probability of being better" doesn't tell you by how much

**Python:**
```python
import numpy as np
from scipy import stats

def bayesian_ab_test(conversions_a, total_a, conversions_b, total_b,
                     prior_alpha=1, prior_beta=1, n_simulations=100000):
    """Bayesian A/B test for proportions using Beta-Binomial model."""
    
    # Posterior distributions
    posterior_a = stats.beta(prior_alpha + conversions_a, 
                            prior_beta + total_a - conversions_a)
    posterior_b = stats.beta(prior_alpha + conversions_b, 
                            prior_beta + total_b - conversions_b)
    
    # Monte Carlo: simulate and compare
    samples_a = posterior_a.rvs(n_simulations)
    samples_b = posterior_b.rvs(n_simulations)
    
    prob_b_better = np.mean(samples_b > samples_a)
    expected_lift = np.mean((samples_b - samples_a) / samples_a)
    
    # Risk (expected loss if choosing B and it's actually worse)
    losses_b = np.where(samples_a > samples_b, samples_a - samples_b, 0)
    expected_loss_b = np.mean(losses_b)
    
    return {
        'prob_b_better': prob_b_better,
        'expected_lift': expected_lift,
        'expected_loss_b': expected_loss_b,
        'posterior_mean_a': posterior_a.mean(),
        'posterior_mean_b': posterior_b.mean(),
    }

result = bayesian_ab_test(120, 1000, 145, 1000)
print(f"P(B > A): {result['prob_b_better']:.2%}")
print(f"Expected lift: {result['expected_lift']:.2%}")
print(f"Expected loss of choosing B: {result['expected_loss_b']:.4f}")
```

---

## Diagnostic & Validation Tests

These tests aren't used to compare A/B groups directly, but are essential for **validating assumptions** and ensuring your primary test is reliable.

---

### 15. Shapiro-Wilk Test (Normality Check)

**What it is:** Tests whether a sample comes from a normally distributed population. Important for deciding whether to use a parametric test (t-test, ANOVA) or a non-parametric alternative.

**Null hypothesis:** The data is normally distributed.

**When to use:**
- Before deciding between t-test vs. Mann-Whitney U
- Before running ANOVA
- Checking residuals in regression-based A/B analysis

**Interpretation:**
- $p > 0.05$: Fail to reject normality → parametric test is appropriate
- $p < 0.05$: Reject normality → consider non-parametric test

> **Practical note:** For large $n$ (>5000), this test almost always rejects normality even for near-normal data. In practice, use **visual inspection** (Q-Q plot, histogram) alongside this test. For large samples, CLT ensures the t-test is robust even with non-normal data.

**Python:**
```python
from scipy import stats

stat, p_value = stats.shapiro(data)
print(f"Shapiro-Wilk statistic: {stat:.4f}, p-value: {p_value:.4f}")
if p_value > 0.05:
    print("→ No evidence against normality, parametric test OK")
else:
    print("→ Data appears non-normal, consider non-parametric test")
```

---

### 16. Levene's Test (Equal Variance Check)

**What it is:** Tests whether two or more groups have **equal variances** (homoscedasticity). This is a key assumption of the equal-variance t-test and standard ANOVA.

**Null hypothesis:** All groups have equal variance.

**When to use:**
- Before choosing between the equal-variance t-test and Welch's t-test
- Before standard ANOVA (vs. Welch's ANOVA)

**Interpretation:**
- $p > 0.05$: Fail to reject equal variances → equal-variance t-test or standard ANOVA is appropriate
- $p < 0.05$: Reject equal variances → use Welch's t-test or Welch's ANOVA

> **Practical note:** In practice, just default to Welch's t-test. It works nearly as well when variances are equal and much better when they're not. Running Levene's test first adds unnecessary complexity.

**Python:**
```python
from scipy import stats

stat, p_value = stats.levene(group_a, group_b)
print(f"Levene's statistic: {stat:.4f}, p-value: {p_value:.4f}")
if p_value > 0.05:
    print("→ Equal variances likely, standard t-test OK")
else:
    print("→ Unequal variances, use Welch's t-test")
```

---

### 17. Kolmogorov-Smirnov Test (Distribution Comparison)

**What it is:** Tests whether two samples come from the **same distribution** (two-sample KS test), or whether a sample matches a specific distribution (one-sample KS test). It measures the maximum distance between the two empirical CDFs.

**Test statistic:**

$$D = \max_x |F_A(x) - F_B(x)|$$

Where $F_A(x)$ and $F_B(x)$ are the empirical cumulative distribution functions of groups A and B.

**When to use:**
- Testing whether the entire **distribution** of a metric differs (not just the mean)
- Detecting differences in shape, spread, or location
- As a complement to mean-based tests — two groups can have the same mean but different distributions

**Advantages:**
- ✅ Sensitive to any type of distributional difference (location, spread, shape)
- ✅ Non-parametric
- ✅ Works as a general "are these different?" test

**Disadvantages:**
- ❌ Less powerful for detecting specific differences (e.g., just the mean)
- ❌ Sensitive to sample size — large samples detect tiny, irrelevant differences
- ❌ Not great for discrete data (designed for continuous distributions)

**Python:**
```python
from scipy import stats

ks_stat, p_value = stats.ks_2samp(group_a, group_b)
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
```

---

## Comparison Summary

### By Metric Type

| Metric Type | Primary Test | Alternative | Multi-Group |
|-------------|-------------|-------------|-------------|
| **Binary / Proportion** (CTR, conversion) | Z-test | Fisher's Exact (small $n$) | Chi-Square |
| **Continuous, normal** (avg order value) | Welch's t-test | Two-sample t-test | ANOVA / Welch's ANOVA |
| **Continuous, skewed** (revenue, time) | Mann-Whitney U | Bootstrap | Kruskal-Wallis |
| **Ordinal** (ratings 1-5) | Mann-Whitney U | Wilcoxon signed-rank (paired) | Kruskal-Wallis |
| **Categorical** (plan choices) | Chi-Square | Fisher's Exact (small $n$) | Chi-Square |
| **Custom / complex** (ratios, percentiles) | Permutation Test | Bootstrap | — |

### By Sample Size

| Sample Size | Proportions | Continuous |
|------------|-------------|------------|
| **Small** ($n < 30$) | Fisher's Exact | Mann-Whitney U or Permutation |
| **Medium** ($30 \leq n < 1000$) | Z-test | Welch's t-test |
| **Large** ($n > 1000$) | Z-test | Welch's t-test (or Bootstrap for skewed data) |

### Complete Test Reference Table

| # | Test | Type | Data | Groups | Normality | Equal Var | Sample Size | Output |
|---|------|------|------|--------|-----------|-----------|-------------|--------|
| 1 | Z-test (proportions) | Parametric | Binary | 2 | — | — | Large ($n>30$) | Z-stat, p-value |
| 2 | Two-sample t-test | Parametric | Continuous | 2 | Yes | **Yes** | Any | t-stat, p-value |
| 3 | Welch's t-test | Parametric | Continuous | 2 | Yes | **No** | Any | t-stat, p-value |
| 4 | One-way ANOVA | Parametric | Continuous | 3+ | Yes | Yes | Any | F-stat, p-value |
| 5 | Paired t-test | Parametric | Continuous | 2 (paired) | Yes | — | Any | t-stat, p-value |
| 6 | Pearson correlation | Parametric | Continuous | 2 variables | Yes | — | Any | r, p-value |
| 7 | Chi-square | Non-parametric | Categorical | 2+ | — | — | Expected freq $\geq 5$ | $\chi^2$, p-value |
| 8 | Fisher's exact | Non-parametric | Categorical | 2 (2x2) | — | — | Any (esp. small) | OR, p-value |
| 9 | Mann-Whitney U | Non-parametric | Ordinal/Continuous | 2 | No | — | Any | U-stat, p-value |
| 10 | Kruskal-Wallis | Non-parametric | Ordinal/Continuous | 3+ | No | — | Any | H-stat, p-value |
| 11 | Wilcoxon signed-rank | Non-parametric | Ordinal/Continuous | 2 (paired) | No | — | Any | W-stat, p-value |
| 12 | Permutation test | Resampling | Any | 2+ | No | No | Any | p-value |
| 13 | Bootstrap test | Resampling | Any | 2+ | No | No | Any | CI, p-value |
| 14 | Bayesian (Beta-Binomial) | Bayesian | Binary | 2+ | — | — | Any | P(B>A), CI |
| 15 | Shapiro-Wilk | Diagnostic | Continuous | 1 | — | — | $n < 5000$ | W-stat, p-value |
| 16 | Levene's test | Diagnostic | Continuous | 2+ | — | — | Any | F-stat, p-value |
| 17 | Kolmogorov-Smirnov | Diagnostic | Continuous | 2 | — | — | Any | D-stat, p-value |

---

## Industry Best Practices

### What the Top Tech Companies Use

| Company | Primary Approach | Notable Technique |
|---------|-----------------|-------------------|
| **Google** | Frequentist + Permutation | CUPED for variance reduction |
| **Microsoft** | Frequentist (t-test) | Trustworthy Online Controlled Experiments |
| **Netflix** | Frequentist + Bootstrap | Fixed-horizon with bootstrap CIs |
| **Uber** | Bayesian + Frequentist | Bayesian for continuous monitoring |
| **Airbnb** | Frequentist + Bootstrap | Bootstrap for complex metrics |
| **Booking.com** | Frequentist (Z-test, t-test) | Sequential testing for early stopping |

### Rules of Thumb for Practitioners

1. **Default to Welch's t-test** for continuous metrics — there is almost never a reason to use the equal-variance t-test.
2. **Use Z-test for proportions** for binary metrics — it's the industry standard.
3. **Use bootstrap** when your metric is complex (ratios, percentiles) or when data is heavily skewed.
4. **Use permutation tests** when you want a distribution-free p-value for any metric.
5. **Always check practical significance** — a statistically significant result may be too small to matter.
6. **Don't peek without correction** — if you monitor results during the test, use sequential testing methods to avoid inflated false positive rates.
7. **Report confidence intervals**, not just p-values — they tell you the likely range of the true effect.
