---
title: "PermutationTestResult<T>"
description: "Represents the results of a permutation test, which is a non-parametric statistical significance test that determines whether the observed difference between two groups is statistically significant."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the results of a permutation test, which is a non-parametric statistical significance test that determines
whether the observed difference between two groups is statistically significant.

## For Beginners

This class stores the results of a permutation test, which helps determine if a difference between groups is real or could have happened by chance.

For example, you might use this test to answer questions like:

- Is the difference in treatment outcomes between two groups statistically significant?
- Could the observed difference in test scores between teaching methods have occurred by random chance?
- Is the relationship between two variables stronger than would be expected by chance?

The test works by:

1. Calculating the actual difference between your groups
2. Randomly shuffling your data many times and recalculating the difference each time
3. Seeing how often the random shuffles produce a difference as extreme as your actual difference

This approach is particularly useful when:

- You have small sample sizes
- Your data doesn't follow a normal distribution
- You want to avoid making assumptions about the underlying distribution

This class stores all the information about the test results, helping you interpret whether
the observed difference is statistically significant.

## How It Works

The permutation test is a resampling method that creates a reference distribution by randomly reassigning observations 
to groups and recalculating the test statistic many times. This class stores the results of such a test, including 
the observed difference between groups, the p-value, the number of permutations performed, the count of extreme values, 
and whether the result is statistically significant. Permutation tests are particularly useful when the assumptions of 
parametric tests (like t-tests) are not met, or when dealing with small sample sizes. The class uses generic type 
parameter T to support different numeric types for the statistical values, such as float, double, or decimal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PermutationTestResult(,,Int32,Int32,)` | Initializes a new instance of the PermutationTestResult class with the specified test statistics and parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CountExtremeValues` | Gets or sets the count of permutations that resulted in a difference as extreme as, or more extreme than, the observed difference. |
| `IsSignificant` | Gets or sets a value indicating whether the permutation test result is statistically significant. |
| `ObservedDifference` | Gets or sets the observed difference between the two groups being compared. |
| `PValue` | Gets or sets the p-value associated with the permutation test. |
| `Permutations` | Gets or sets the number of permutations performed during the test. |
| `SignificanceLevel` | Gets or sets the significance level used for the test. |

