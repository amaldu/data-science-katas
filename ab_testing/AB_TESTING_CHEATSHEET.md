# A/B Testing Cheatsheet

> A complete reference for designing, running, and analyzing A/B tests in data science interviews and real-world practice.

---

## Table of Contents

1. [What is A/B Testing?](#1-what-is-ab-testing)
2. [End-to-End Workflow](#2-end-to-end-workflow)
3. [Step 1 - Understand the Business](#step-1---understand-the-business)
4. [Step 2 - Define Hypotheses](#step-2---define-hypotheses)
5. [Step 3 - Select Metrics](#step-3---select-metrics)
6. [Step 4 - Design the Experiment](#step-4---design-the-experiment)
7. [Step 5 - Validity Checks](#step-5---validity-checks)
8. [Step 6 - Run the Test](#step-6---run-the-test)
9. [Step 7 - Analyze Results](#step-7---analyze-results)
10. [Step 8 - Make a Decision](#step-8---make-a-decision)
11. [Statistical Tests Reference](#statistical-tests-reference)
12. [Advanced Topics](#advanced-topics)
13. [Common Pitfalls](#common-pitfalls)
14. [Quick Decision Flowchart](#quick-decision-flowchart)

---

## 1. What is A/B Testing?

A/B testing (also called split testing or controlled experimentation) is a method of comparing **two or more versions** of a product, feature, or experience to determine which one performs better on a specific metric.

| Term | Definition |
|------|-----------|
| **Control (A)** | The existing/baseline version |
| **Treatment (B)** | The new version being tested |
| **Variant** | Any version in the test (A, B, C...) |
| **Randomization Unit** | The entity being randomly assigned (usually users) |
| **Primary Metric** | The main KPI used to evaluate the test |
| **Guardrail Metrics** | Secondary metrics monitored to ensure no harm |

---

## 2. End-to-End Workflow

```
Business Understanding --> Define Hypotheses --> Select Metrics --> Design Experiment
        |                                                                |
        v                                                                v
  Make Decision  <--  Analyze Results  <--  Run the Test  <--  Validity Checks
```

---

## Step 1 - Understand the Business

Before anything else, understand the **full context**:

- Navigate the core product and understand the **user journey**
- Identify the **funnel stages** and where the change will be applied
- Talk to stakeholders about **business goals** and constraints
- Understand what **success looks like** from a business perspective

> **Interview tip:** Always start by asking clarifying questions about the business context before jumping into statistical design.

---

## Step 2 - Define Hypotheses

### Null Hypothesis ($H_0$)
There is **no significant difference** between control and treatment.

### Alternative Hypothesis ($H_a$)
There **is a significant difference** between control and treatment.

### Types of Tests

| Type | $H_a$ | When to Use |
|------|--------|-------------|
| **Two-tailed** | $\mu_A \neq \mu_B$ | You want to detect any difference (positive or negative) |
| **One-tailed (right)** | $\mu_B > \mu_A$ | You only care if treatment is better |
| **One-tailed (left)** | $\mu_B < \mu_A$ | You only care if treatment is worse |

> **Rule of thumb:** Use **two-tailed tests** unless you have a strong prior reason to test in one direction only.

### PICOT Framework for Hypothesis Design

| Letter | Component | Example |
|--------|-----------|---------|
| **P** | Population | Users who reach checkout page |
| **I** | Intervention | New checkout button design |
| **C** | Comparison | Control group sees old design |
| **O** | Outcome | Conversion rate difference |
| **T** | Time | 2-week experiment window |

---

## Step 3 - Select Metrics

### Primary Metric (Decision Metric)

The single metric used to decide if the test is successful. It must be:

| Property | Description |
|----------|-------------|
| **Measurable** | Quantifies a specific user behavior |
| **Attributable** | Direct causal relationship between change and metric |
| **Sensitive** | Low inherent variability; can distinguish between groups |
| **Timely** | Captures behavior within the test window |

**Common Primary Metrics:**
- Conversion rate, click-through rate (CTR), revenue per user, time on page, retention rate

### Guardrail Metrics

Metrics monitored to **ensure no negative side effects**:
- Page load time (latency)
- Bounce rate
- Revenue (if not the primary metric)
- Error rates
- User complaints / support tickets

### OEC (Overall Evaluation Criterion)

A composite metric that combines multiple signals into a single score. Useful when no single metric captures the full picture.

---

## Step 4 - Design the Experiment

### 4.1 Significance Level ($\alpha$)

$$
\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true}) = P(\text{Type I Error})
$$

| Context | Typical $\alpha$ |
|---------|-----------------|
| Standard A/B test | 0.05 (5%) |
| High-risk change (e.g., pricing) | 0.01 (1%) |
| Exploratory / low-risk | 0.10 (10%) |

### 4.2 Statistical Power ($1 - \beta$)

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_0 \text{ is false})
$$

| Term | Definition |
|------|-----------|
| $\beta$ | Probability of Type II Error (false negative) |
| Power | Probability of detecting a real effect |
| Typical value | 0.80 (80%) |

### 4.3 Minimum Detectable Effect (MDE / $\delta$)

The **smallest effect size** that is meaningful from a business perspective.

| Platform Size | Typical MDE |
|--------------|-------------|
| Large (millions of users) | 0.5% - 1% |
| Medium | 2% - 5% |
| Small / early stage | 5% - 10% |

> **Trade-off:** Smaller MDE = larger sample size = longer test duration.

### 4.4 Type I vs Type II Errors

| | $H_0$ is True | $H_0$ is False |
|---|---|---|
| **Reject $H_0$** | Type I Error ($\alpha$) | Correct (Power) |
| **Fail to reject $H_0$** | Correct | Type II Error ($\beta$) |

### 4.5 Sample Size Calculation

**Rule of thumb** (for $\alpha = 0.05$, Power = 0.80):

$$
n \approx \frac{16 \cdot \sigma^2}{\delta^2}
$$

**For proportions** (e.g., conversion rates):

$$
n = \frac{(Z_{\alpha/2} + Z_\beta)^2 \cdot [p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}
$$

Where:
- $Z_{\alpha/2}$ = 1.96 (for $\alpha$ = 0.05, two-tailed)
- $Z_\beta$ = 0.84 (for Power = 0.80)
- $p_1, p_2$ = expected proportions in control and treatment

### 4.6 Test Duration

$$
\text{Duration (days)} = \frac{\text{Required sample size per group} \times 2}{\text{Daily eligible visitors}}
$$

**Guidelines:**
- **Minimum:** 1 week (to capture weekday/weekend patterns)
- **Typical:** 2-4 weeks
- **Maximum:** Avoid running longer than necessary (opportunity cost)

**Effects to watch for:**

| Effect | Description | Mitigation |
|--------|-------------|------------|
| **Novelty Effect** | Users react positively to anything new, effect fades | Run test longer; segment by new vs returning users |
| **Primacy Effect** | Users prefer what they're used to, resistance to change fades | Run test longer |
| **Maturation Effect** | User behavior stabilizes as they get familiar | Account for learning curve |
| **Seasonality** | Behavior varies by day/week/month/holiday | Avoid holidays; run full weeks |

### 4.7 Traffic Split

| Split | When to Use |
|-------|-------------|
| **50/50** | Standard - maximizes statistical power |
| **90/10 or 95/5** | High-risk changes; minimize exposure |
| **Unequal splits** | When you want to limit risk but still get signal |

---

## Step 5 - Validity Checks (NESSI)

Run these checks **before analyzing results** to ensure experiment integrity:

| Check | Method | What it Detects |
|-------|--------|----------------|
| **N**ovelty Effect | Segment by new vs. returning users | Temporary behavior change |
| **E**xternal Factors | Check for holidays, outages, competitor actions | Confounding events |
| **S**ample Ratio Mismatch (SRM) | Chi-square goodness-of-fit test | Broken randomization |
| **S**election Bias | A/A test or pre-experiment metric comparison | Non-random group assignment |
| **I**nstrumentation | Monitor guardrail metrics (latency, errors) | Technical issues affecting results |

### Sample Ratio Mismatch (SRM) - Critical Check

If you expected 50/50 split and got 52/48, run a chi-square test:

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

If $p < 0.01$, there is likely a **bug in the randomization** - do NOT trust the results.

---

## Step 6 - Run the Test

- **Do not peek** at results before the planned end date (peeking inflates false positive rate)
- Monitor **guardrail metrics** for safety
- Have a **stop criterion** for catastrophic failures
- Log all data properly for post-analysis

---

## Step 7 - Analyze Results

### 7.1 Calculate Test Statistic

For a **Z-test on proportions:**

$$
Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}
$$

Where $\hat{p} = \frac{x_A + x_B}{n_A + n_B}$ is the pooled proportion.

For a **t-test on means:**

$$
t = \frac{\bar{X}_B - \bar{X}_A}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}
$$

### 7.2 Calculate p-value

The p-value is the probability of observing data as extreme as (or more extreme than) the observed results, assuming $H_0$ is true.

| p-value | Interpretation |
|---------|---------------|
| $p < \alpha$ | **Reject** $H_0$ - result is statistically significant |
| $p \geq \alpha$ | **Fail to reject** $H_0$ - no significant difference detected |

> **Important:** "Fail to reject" $\neq$ "Accept $H_0$". Absence of evidence is not evidence of absence.

### 7.3 Confidence Interval

$$
CI = (\hat{p}_B - \hat{p}_A) \pm Z_{\alpha/2} \cdot SE
$$

The confidence interval tells you the **range of plausible effect sizes**. If it contains 0, the result is not significant.

### 7.4 Practical vs Statistical Significance

| | Statistically Significant | Not Statistically Significant |
|---|---|---|
| **Practically Significant** | Launch the change | Increase sample size or reconsider |
| **Not Practically Significant** | Effect is real but too small to matter | No action needed |

---

## Step 8 - Make a Decision

| Scenario | Action |
|----------|--------|
| Significant + practically meaningful | **Launch** treatment |
| Significant but trivially small effect | Consider cost/benefit; likely **don't launch** |
| Not significant, CI excludes meaningful effect | **Don't launch** - no effect detected |
| Not significant, CI includes meaningful effect | **Inconclusive** - need more data |
| Guardrail metrics degraded | **Don't launch** regardless of primary metric |

---

## Statistical Tests Reference

### Parametric Tests

| Test | Data Type | Groups | Key Assumption | When to Use in A/B Testing |
|------|-----------|--------|----------------|---------------------------|
| **Z-test (proportions)** | Binary / Proportions | 2 | $n > 30$, known variance | Comparing conversion rates, CTR |
| **Two-sample t-test** | Continuous | 2 | Normal distribution, equal variance | Comparing means (revenue, time) with small $n$ |
| **Welch's t-test** | Continuous | 2 | Normal distribution, **unequal variance** | Default t-test when variances differ |
| **ANOVA** | Continuous | 3+ | Normal distribution, equal variance | A/B/n tests comparing multiple variants |
| **Welch's ANOVA** | Continuous | 3+ | Normal distribution, **unequal variance** | A/B/n tests when variances differ |
| **Pearson's Correlation** | Continuous | 2 variables | Linearity, normality | Measuring linear relationship between metrics |

### Non-Parametric Tests

| Test | Data Type | Groups | When to Use in A/B Testing |
|------|-----------|--------|---------------------------|
| **Chi-square test** | Categorical | 2+ | Comparing proportions with sufficient sample ($\geq 5$ expected per cell) |
| **Fisher's Exact Test** | Categorical | 2 | Small samples (expected frequency $< 5$ in any cell) |
| **Mann-Whitney U** | Ordinal / Continuous | 2 | Non-normal data (ratings, skewed revenue) |
| **Kruskal-Wallis** | Ordinal / Continuous | 3+ | Non-normal data with 3+ groups |

### Detailed Test Comparison

#### Z-test for Proportions
- **Advantages:** Simple, fast, well-understood; works great for large samples
- **Disadvantages:** Requires large $n$; assumes known population variance
- **A/B use case:** Conversion rate comparison between control and treatment

#### Two-Sample t-test
- **Advantages:** Works with smaller samples; doesn't require known population variance
- **Disadvantages:** Assumes equal variance and normality; sensitive to outliers
- **A/B use case:** Average order value, session duration comparison

#### Welch's t-test
- **Advantages:** Robust to unequal variances; no equal variance assumption needed
- **Disadvantages:** Still assumes normality; less power than equal-variance t-test when variances are actually equal
- **A/B use case:** Preferred default over standard t-test in practice

#### Chi-Square Test
- **Advantages:** Handles categorical data with multiple categories; easy to compute
- **Disadvantages:** Requires expected frequency $\geq 5$ per cell; doesn't show direction of effect
- **A/B use case:** Comparing click distributions, purchase categories

#### Fisher's Exact Test
- **Advantages:** Exact (not approximate); works with very small samples
- **Disadvantages:** Computationally expensive for large tables; limited to 2x2 in practice
- **A/B use case:** Small pilot tests, rare conversion events

#### Mann-Whitney U Test
- **Advantages:** No normality assumption; robust to outliers; works with ordinal data
- **Disadvantages:** Less powerful than t-test when normality holds; tests distributions, not just means
- **A/B use case:** User ratings, skewed revenue data, non-normal metrics

#### ANOVA / Kruskal-Wallis
- **Advantages:** Compares 3+ groups simultaneously; controls family-wise error rate
- **Disadvantages:** Only tells you groups differ, not which ones (need post-hoc tests like Tukey HSD)
- **A/B use case:** Multi-variant tests (A/B/C/D)

---

## Advanced Topics

### Bayesian A/B Testing

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Core idea** | Fixed parameters, random data | Random parameters, fixed data |
| **Result** | p-value, confidence interval | Posterior distribution, credible interval |
| **Interpretation** | "Probability of data given $H_0$" | "Probability that B is better than A" |
| **Sample size** | Fixed upfront | Can stop anytime (with caveats) |
| **Prior knowledge** | Not incorporated | Can incorporate prior beliefs |
| **Decision** | Reject / fail to reject | P(B > A), expected loss |

**Bayesian approach for proportions:**
- Prior: $p \sim \text{Beta}(\alpha, \beta)$
- Likelihood: $x \sim \text{Binomial}(n, p)$
- Posterior: $p \mid x \sim \text{Beta}(\alpha + x, \beta + n - x)$

**When to use Bayesian:**
- You want interpretable probabilities ("82% chance B is better")
- You want to incorporate prior knowledge
- You need flexibility to stop early
- Business prefers "probability of being better" over p-values

### Sequential Testing

Unlike fixed-horizon tests, sequential testing allows **peeking at results** while controlling error rates:

- **Group Sequential Testing:** Pre-planned interim analyses with adjusted significance thresholds (e.g., O'Brien-Fleming, Pocock bounds)
- **Always-Valid p-values:** Methods that maintain Type I error control regardless of when you look

### Multiple Testing Correction

When running **multiple tests simultaneously**, the probability of at least one false positive increases:

$$
P(\text{at least 1 false positive}) = 1 - (1 - \alpha)^k
$$

| Method | Approach | Strictness |
|--------|----------|------------|
| **Bonferroni** | Divide $\alpha$ by number of tests ($\alpha / k$) | Most conservative |
| **Holm-Bonferroni** | Step-down procedure | Less conservative than Bonferroni |
| **Benjamini-Hochberg (FDR)** | Controls false discovery rate | Least conservative |

### CUPED (Controlled-experiment Using Pre-Experiment Data)

Variance reduction technique that uses **pre-experiment data** to reduce noise:

$$
\hat{Y}_{cv} = \hat{Y} - \theta(\hat{X} - \bar{X})
$$

Where $X$ is the pre-experiment covariate and $\theta = \frac{Cov(Y, X)}{Var(X)}$.

**Benefits:** Can reduce variance by 50%+ and significantly decrease required sample size.

### Multi-Armed Bandit (MAB)

An **adaptive** alternative to classical A/B testing:

| Aspect | A/B Test | Multi-Armed Bandit |
|--------|----------|-------------------|
| **Traffic allocation** | Fixed split | Dynamic (more traffic to winner) |
| **Regret** | Higher (equal allocation) | Lower (exploits best variant) |
| **Statistical rigor** | Higher | Lower (harder to get clean inference) |
| **Best for** | Definitive answers | Optimization with limited time |

**Common algorithms:** Epsilon-Greedy, UCB (Upper Confidence Bound), Thompson Sampling

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Peeking** | Checking results repeatedly inflates false positives | Use sequential testing or wait for planned end date |
| **Small sample** | Underpowered test misses real effects | Calculate sample size before running |
| **Too many metrics** | Multiple comparisons increase false positives | Choose one primary metric; apply corrections |
| **Selection bias** | Non-random assignment biases results | Verify randomization with A/A tests and SRM checks |
| **Survivorship bias** | Only analyzing users who completed an action | Define analysis population before seeing results |
| **Simpson's paradox** | Aggregate trend reverses in subgroups | Segment analysis; check for confounders |
| **Network effects** | Treatment affects control through social connections | Use cluster randomization |
| **Interference** | Users in different groups interact | Use proper randomization units |
| **HARKing** | Hypothesizing After Results are Known | Pre-register hypotheses and metrics |

---

## Quick Decision Flowchart

```
What type of metric are you comparing?
│
├── Binary / Proportion (e.g., conversion rate, CTR)
│   ├── Sample size ≥ 30 per group? ──> Z-test for proportions
│   ├── Expected freq ≥ 5 per cell? ──> Chi-Square test
│   └── Small sample (freq < 5)?   ──> Fisher's Exact Test
│
├── Continuous (e.g., revenue, time on page)
│   ├── Normal distribution?
│   │   ├── 2 groups, equal variance?   ──> Two-sample t-test
│   │   ├── 2 groups, unequal variance? ──> Welch's t-test
│   │   └── 3+ groups?                  ──> ANOVA (+ post-hoc)
│   └── Non-normal distribution?
│       ├── 2 groups? ──> Mann-Whitney U test
│       └── 3+ groups? ──> Kruskal-Wallis test
│
├── Categorical (e.g., plan type, button clicked)
│   └── Chi-Square test (or Fisher's if small sample)
│
└── Want probability of B > A instead of p-value?
    └── Bayesian A/B Testing
```

---

## Python Quick Reference

```python
# Z-test for proportions
from statsmodels.stats.proportion import proportions_ztest
stat, pval = proportions_ztest([successes_A, successes_B], [n_A, n_B])

# Chi-square test
from scipy.stats import chi2_contingency
chi2, pval, dof, expected = chi2_contingency([[a_success, a_fail], [b_success, b_fail]])

# Two-sample t-test
from scipy.stats import ttest_ind
stat, pval = ttest_ind(group_a, group_b)

# Welch's t-test
stat, pval = ttest_ind(group_a, group_b, equal_var=False)

# Mann-Whitney U
from scipy.stats import mannwhitneyu
stat, pval = mannwhitneyu(group_a, group_b, alternative='two-sided')

# Fisher's Exact Test
from scipy.stats import fisher_exact
odds_ratio, pval = fisher_exact([[a_success, a_fail], [b_success, b_fail]])

# ANOVA
from scipy.stats import f_oneway
stat, pval = f_oneway(group_a, group_b, group_c)

# Sample size calculation
from statsmodels.stats.power import NormalIndPower
analysis = NormalIndPower()
n = analysis.solve_power(effect_size=0.05, alpha=0.05, power=0.8, alternative='two-sided')
```

---
