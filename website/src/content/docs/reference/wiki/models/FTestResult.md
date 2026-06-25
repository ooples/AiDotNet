---
title: "FTestResult<T>"
description: "Represents the results of an F-test, which is used to compare the variances of two populations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the results of an F-test, which is used to compare the variances of two populations.

## For Beginners

The F-test helps determine if two groups have similar or different amounts of variability.

For example, you might use this test to answer questions like:

- Do men and women have the same variability in test scores?
- Is the precision of one measurement method better than another?
- Are the variances in two manufacturing processes comparable?

The test works by:

1. Calculating the ratio of the two sample variances
2. Comparing this ratio to what would be expected if the population variances were equal
3. Determining if the difference is statistically significant

This class stores all the information about the test results, helping you interpret whether
the observed difference in variability is likely due to chance or represents a real difference.

## How It Works

The F-test is a statistical test used to determine whether two populations have equal variances. It is based on 
the ratio of two sample variances. This class stores the results of such a test, including the F-statistic, 
p-value, degrees of freedom, the variances being compared, confidence intervals, and whether the result is 
statistically significant. The F-test is commonly used in analysis of variance (ANOVA) and as a preliminary 
test before applying other statistical tests that assume equal variances. The class uses generic type parameter 
T to support different numeric types for the statistical values, such as float, double, or decimal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FTestResult(,,Int32,Int32,,,,,)` | Initializes a new instance of the FTestResult class with the specified test statistics and parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DenominatorDegreesOfFreedom` | Gets or sets the degrees of freedom for the denominator. |
| `FStatistic` | Gets or sets the F-statistic value. |
| `IsSignificant` | Gets or sets a value indicating whether the F-test result is statistically significant. |
| `LeftVariance` | Gets or sets the variance of the left (or first) sample. |
| `LowerConfidenceInterval` | Gets or sets the lower bound of the confidence interval for the ratio of population variances. |
| `NumeratorDegreesOfFreedom` | Gets or sets the degrees of freedom for the numerator. |
| `PValue` | Gets or sets the p-value associated with the F-test. |
| `RightVariance` | Gets or sets the variance of the right (or second) sample. |
| `SignificanceLevel` | Gets or sets the significance level used for the test. |
| `UpperConfidenceInterval` | Gets or sets the upper bound of the confidence interval for the ratio of population variances. |

