# A/B Testing Cheatsheet


<div align="center">
  <img src="images/steps.png" alt="Conversion Rate" width="500" />
</div>


# Steps of the A/B test

## Before anything else... Understanding the business

The most important part is to understand the whole context of the problem. This is pretty obvious but it's important to keep it in mind because can give us a lot of information about how to tackle the next steps. A good approach to this is to first navigate through the core products of the site and create an outline of what that user journey is.

## 1. Defining the statistical hypothesis

We have to think of a solution that could affect the KPIs so we need to compare 2 products or versions or the same item. We set two hypothesis:

### Null hypothesis ($H_0$):
This hypothesis says that **there is no difference between the original version and the new solution**.

### Alternative hypothesis ($H_a$):
This hypothesis says that **there is a significant difference between the original version and the new solution**. In order to define a proper alternative hypothesis it's recommended to use the **PICOT criteria**:

<div align="center">
  <img src="images/picot.jpg" alt="Conversion Rate" width="500" />
</div>

Here is an example of how to implement the PICOT criteria:

 1. **Population**: the users that reach the level of the funnel where our change is made.
 2. **Intervention**: change in the title of the page to optimize engaging.
 3. **Comparison**: set two samples of users, one sees the intervention and the other sees the old version.
 4. **Outcome**: the results of the comparison (see below)
 5. **Time**: the time we need to set to perform a good A/B test (see below)

## 2. Selection of the primary metric
The metric's choice depends on the underlying hipothesis that is being tested and **has to increase significantly while everything else stays constant**. 
Meanwhile you perform the A/B test you will have to keep an eye to other metrics to ensure that they don't change much and the test is valid.

The metrics have to be measurable (quantifies a behaviour), attributable (there is a direct relationship between cause and effect), sensitive (shows low variability per se so you can distinguish both groups) and timely (measures behavior in  short term)

The metric is a business metric. Just remember, they have to represent ONE behaviour that doesn't affect the rest. 

## 3. Design of the experiment
In the experiment, we aim to avoid two types of errors:

**Type I Error (False Positives)**    
We define the variable *α* as the probability of making a Type I Error (rejecting the null hypothesis when it is true).

**Type II Error (False Negatives)**
We define the variable *β* as the probability of making a Type II Error (failing to reject the null hypothesis when it is false).

To design the experiment, we will follow these steps:

### 3.1. Determine the significance level of the test

The significance level (α) represents the probability of rejecting the null hypothesis when it is actually true, also known as the Type I Error rate.

$$
\text{Significance Level} =  \alpha = \text{probability of making Type I Error}
$$

In A/B testing, a common value for the significance level is 5%, meaning we accept a 5% chance of falsely detecting a treatment effect when no real effect exists. This corresponds to a 95% confidence level in observing a significant difference between the two groups.

The choice of the significance level depends on the nature of the test and the associated business constraints.


### 3.2. Determine the power of the test power

Power is the probability of making a correct decision (rejecting the null hypothesis) when the null hypothesis is false.

$$
\text{Power} = 1 - \beta = \text{probability of not making Type II Error}
$$

A common value for the power of an A/B test is 80%, which corresponds to a 20% Type II Error rate. This means we are willing to accept a 20% chance of failing to detect a treatment effect when one actually exists.

However, the power value should be tailored to the specific context of the test and the associated business constraints.

### 3.3. Determine the Minimum Detectable Effect

The Minimum Detectable Effect (MDE or δ) represents the smallest effect between the experimental groups that would be considered impactful from a business perspective. 


This threshold is typically set by the business stakeholders based on their objectives, constraints, and the potential value of implementing the new version. A common value for a large online platform with millions of users is 1%


### 3.4. Set the randomization unit.

The randomization unit is the unit we are going to measure and select randomly. An example can be the user of the site.

### 3.5. Target the population of the experiment
Among all the users that use our site we have to define which ones are going to participate in the experiment. In the example of a funnel, the users that would reach a specific level of the funnel.

### 3.6. Determine the sample size

A rule of thumb for the sample size (\( n \)) can be estimated using:

\[
n \approx \frac{16 \cdot \sigma^2}{\delta^2}
\]

Where:
- \( \sigma \) = standard deviation of the metric  
- \( \delta \) = minimum detectable difference (effect size)  

taking into consideration two conditions that are common in the industry:
 1. alpha = 0.05
 2. Power of the test = 0.8


### 3.7. Determine the test duration

The baseline duration time of the test is determined by this formula:

$$
\text{duration} = \frac{\text{minimum sample size (N)}}{\text{visitors per day}}
$$

And it's usually between one and two weeks. Less than one week won't take into consideration special days where users can have a different behaviour (like the weekend).

Remember that picking the right period of time in the year and avoid days where the behaviour of the customers can vary significantly due to external reasons.

Effects to take into consideration:

##### 3.7.1 Novelty effect

Users react quickly and positively to types of changes independent of their nature but this behaviour wears off in time. 
The test must take longer than such effects can last otherwise the results won't be valid.

##### 3.7.2 Maturation effect

Users or participants become more familiar with the product, process, or environment over time. As they grow accustomed to the change being tested, their behavior may stabilize, leading to changes in the results.

## 4. Do validity checks
Faulty experiments can lead to a bad decision. Some things we have to check are...(NESSI)

1. Novelty effect: segmenting by new/old visitors first to see if there is a difference between
2. External factors:holidays, competition, economic distruptions, etc.
3. Selection bias (A/A test): are the sample homogeneous?
4. Sample ratio mismatch (chi-square goodness of fit test): to ensure the ratio between two samples is 50-50
5. Instrumentation effect (guardrail metrics): latency time, etc.


## 5. Run de test 

Now it's time to wait...

<p align="center">
  <img src="images/loading-progress-bar.gif" width="300" />
</p>

## 6. Parametric tests


### 6.1 Two-Sample t-test (Independent t-test)

**When to Apply:**    
When the metric follows a t-student distribution and we are comparing the means of two independent groups (e.g., control vs. treatment) of a small sample (typically n < 30).

**Conditions:**
    The data in each group must be normally distributed.
    The groups must be independent.
    Variances between the groups should be approximately equal (this can be tested using Levene’s test).

**Limitations:**
    If the assumption of equal variances is violated, a Welch’s t-test should be used instead.
    The test is sensitive to outliers and extreme values in the data.


### 6.2 Two-sample Z-Test 

**When to Apply:**    
Use when N > 30, metric follows asymptotic Normal distribution and you want to identify whether there exist a relationship and the type of relationship between control and experimental groups.

**Conditions:**    
The sample size should be large enough (n > 30) for the Central Limit Theorem to apply and approximate the distribution of the sample mean to normal.
    The population standard deviation must be known or assumed to be known.
    The data should follow a normal distribution or be sufficiently large for the approximation.

**Limitations:**    
If the sample size is small (n ≤ 30), the t-test is more appropriate because the Z-test assumes a large sample.
    If the population standard deviation is unknown, a t-test should be used instead.


## 7. Non-parametric tests

### 7.1 Fisher’s Exact Test

**When to Apply:**
    Use when analyzing small sample sizes, especially in 2x2 contingency tables, where expected cell frequencies are less than 5.
    Example: A study comparing the success rates of two treatments with small sample sizes (e.g., 3 successes and 2 failures in both treatment groups).

**Conditions:**
    Works with categorical data in 2x2 tables.
    Applied when the expected frequency in any of the cells is below 5. This is a critical assumption for Chi-Square, but Fisher's Exact Test can still provide reliable results.

**Limitations:**
    Computationally intensive for large tables, making it less practical for tables larger than 2x2.
    Limited to small sample sizes and tables with fewer than five expected frequencies.

### 7.2 Chi-Square Test

**When to Apply:**
    Use the Chi-Square Test when comparing observed frequencies to expected frequencies in categorical data, particularly for larger sample sizes.
    Example: In an A/B test, you might use the Chi-Square Test to compare the conversion rates (success vs. failure) between two groups (Group A and Group B). You have enough observations in each group to analyze the differences in proportions.

**Conditions:**
    The data should consist of categorical variables (e.g., success/failure, yes/no).
    The sample size should be large enough for each expected frequency to be at least 5 (this is a key assumption for the Chi-Square test).
    Typically used for 2x2 contingency tables but can be extended to larger tables (more than two categories per variable).
    The observations should be independent (i.e., one observation does not influence another).

**Limitations:**
    The Chi-Square test becomes less accurate when expected cell frequencies are too small (less than 5), which could lead to misleading results. This is particularly a problem in 2x2 tables with small sample sizes, where Fisher's Exact Test may be more appropriate.
    The test assumes that the samples are independent. If the data involves repeated measures or paired observations, this test is not suitable.
    It cannot tell you which group or category differs; it only tells you whether there is a significant difference between observed and expected frequencies. Post-hoc tests might be needed to determine which specific categories are different.
<!-- 
### 7.3 Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

**When to Apply:**
    This is a non-parametric test used to compare the distributions of two independent groups when the data is not normally distributed.
    Example: Comparing customer ratings of two different products where the ratings are ordinal or skewed.

**Conditions:**
    The two groups must be independent (no overlap in observations).
    The test does not require the assumption of normality.
    The data should be at least ordinal, meaning the values have a meaningful order but not necessarily equal intervals.

**Limitations:**
    While it does not require normality, it does assume that the distributions of the two groups are similar in shape. If the distributions are very different, the results may be misleading.
    Less powerful than the independent t-test when data is normally distributed, as it does not take advantage of parametric methods.
 -->


-----

### 5.6 ANOVA

**When to Apply:**
    Use when comparing the means of three or more independent groups.
    Example: Comparing the average salary of employees across different departments (Marketing, Sales, IT).

**Conditions:**
    The groups must be independent.
    The data in each group should be normally distributed.
    The variances among groups should be equal (can be tested using Levene’s test).

**Limitations:**
    Assumes homogeneity of variance. If this assumption is violated, a Welch's ANOVA can be used.
    If the ANOVA test is significant, post-hoc tests (e.g., Tukey HSD) should be performed to determine which groups are different.


### 5.7 Pearson's Correlation Test

**When to Apply:**
    Use the Pearson's Correlation Test to measure the strength and direction of the linear relationship between two continuous variables.
    Example: In an A/B test where you are testing the relationship between the amount of time users spend on your website (in minutes) and the number of pages they visit, you would use Pearson's Correlation to determine if there's a linear relationship between these two variables.

**Conditions:**
    Both variables must be continuous (i.e., not categorical).
    The relationship between the two variables should be linear. You can check for linearity visually using scatterplots.
    Both variables should be approximately normally distributed (for small sample sizes, this assumption becomes more important).
    The data should be free from outliers, as outliers can significantly distort the correlation.

**Limitations:**
    Only measures linear relationships. If the relationship between the variables is non-linear, Pearson's correlation might not be an appropriate test.
    Sensitive to outliers, which can distort the result. Even a few extreme values can significantly affect the correlation coefficient.
    Pearson's correlation does not imply causation. A significant correlation doesn't mean that one variable causes the other.

### 5.8 Welch's t-test

**When to Apply:**
    Use the Welch's t-test when comparing the means of two independent groups, particularly when the assumption of equal variances (homoscedasticity) is violated.
    Example: In an A/B test where you compare the average conversion rates between two groups (Group A and Group B), and you suspect that the variance in conversion rates between the two groups is not equal, you would use Welch's t-test to compare the means of the two groups.

**Conditions:**
    The two groups should be independent.
    The data in each group should be approximately normally distributed (though Welch’s t-test is more robust to deviations from normality than the regular t-test).
    Unequal variances between the two groups. Welch’s t-test is specifically designed to handle situations where the variances of the two groups are different.
    The sample sizes for each group can be unequal, and Welch’s t-test will still provide a valid result.

**Limitations:**
    Although Welch's t-test is robust to unequal variances, it still assumes that the data is independently sampled and that the observations in each group are independent of each other.
    It may not perform as well with very small sample sizes in one or both groups, especially if the normality assumption is violated.
    Like the standard two-sample t-test, it assumes that the data is continuous.

next sections

Calculating the test statistics (T)
Calculating the p-value of the test statistics
Reject or fail to reject the statistical hypothesis (statistical significance)
Calculate the margin of error (external validity of the experiment)
Calculate confidence interval (external validity and practical significance of the experiment)
