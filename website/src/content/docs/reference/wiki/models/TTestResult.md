---
title: "TTestResult<T>"
description: "Represents the results of a t-test, which is a statistical hypothesis test used to determine if there is a significant  difference between the means of two groups."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the results of a t-test, which is a statistical hypothesis test used to determine if there is a significant 
difference between the means of two groups.

## For Beginners

This class stores the results of a t-test, which helps determine if the difference between two groups is statistically significant.

For example, you might use a t-test to answer questions like:

- Is there a significant difference in test scores between two teaching methods?
- Does a new medication significantly change blood pressure compared to a placebo?
- Are the average sales before and after a marketing campaign significantly different?

The t-test works by:

1. Calculating a t-statistic based on the difference between group means
2. Determining how likely this t-statistic would occur by chance
3. Producing a p-value that represents this probability

This test is particularly useful when:

- You're comparing means between two groups
- Your data approximately follows a normal distribution
- You have relatively small sample sizes

This class stores all the information about the test results, helping you interpret whether
the observed difference is statistically significant.

## How It Works

The t-test is one of the most commonly used statistical tests for comparing means. This class stores the results of 
such a test, including the t-statistic, degrees of freedom, p-value, and whether the result is statistically significant. 
The t-test is used when the test statistic follows a t-distribution under the null hypothesis, which typically occurs 
when comparing means from normally distributed populations with unknown variances. The class uses generic type parameter 
T to support different numeric types for the statistical values, such as float, double, or decimal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TTestResult(,Int32,,)` | Initializes a new instance of the TTestResult class with the specified test statistics and parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DegreesOfFreedom` | Gets or sets the degrees of freedom for the t-test. |
| `IsSignificant` | Gets or sets a value indicating whether the t-test result is statistically significant. |
| `PValue` | Gets or sets the p-value associated with the t-test. |
| `SignificanceLevel` | Gets or sets the significance level used for the test. |
| `TStatistic` | Gets or sets the t-statistic value. |

