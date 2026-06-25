---
title: "MannWhitneyUTestResult<T>"
description: "Represents the results of a Mann-Whitney U test, which is a non-parametric statistical test used to determine  whether two independent samples come from the same distribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the results of a Mann-Whitney U test, which is a non-parametric statistical test used to determine 
whether two independent samples come from the same distribution.

## For Beginners

The Mann-Whitney U test helps determine if two groups are different when you can't use a regular t-test.

For example, you might use this test to answer questions like:

- Do patients on treatment A have different recovery times than those on treatment B?
- Do students using one learning method score differently than those using another method?
- Are customer satisfaction ratings different between two product versions?

The test works by:

1. Ranking all values from both groups together
2. Calculating how much the ranks differ between groups
3. Determining if this difference is statistically significant

This test is particularly useful when:

- Your data doesn't follow a normal distribution
- You have ordinal data (rankings) rather than continuous measurements
- You have outliers that might skew the results of a t-test

This class stores all the information about the test results, helping you interpret whether
the observed difference between groups is likely due to chance or represents a real difference.

## How It Works

The Mann-Whitney U test (also known as the Wilcoxon rank-sum test) is a non-parametric alternative to the 
independent samples t-test. It's used when the data doesn't meet the assumptions required for the t-test, 
particularly when the data isn't normally distributed. This class stores the results of such a test, including 
the U statistic, Z-score, p-value, and whether the result is statistically significant. The test is commonly 
used to compare the medians of two groups, though it technically tests whether one sample tends to have larger 
values than the other. The class uses generic type parameter T to support different numeric types for the 
statistical values, such as float, double, or decimal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MannWhitneyUTestResult(,,,)` | Initializes a new instance of the MannWhitneyUTestResult class with the specified test statistics and parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsSignificant` | Gets or sets a value indicating whether the Mann-Whitney U test result is statistically significant. |
| `PValue` | Gets or sets the p-value associated with the Mann-Whitney U test. |
| `SignificanceLevel` | Gets or sets the significance level used for the test. |
| `UStatistic` | Gets or sets the U statistic value. |
| `ZScore` | Gets or sets the Z-score associated with the U statistic. |

