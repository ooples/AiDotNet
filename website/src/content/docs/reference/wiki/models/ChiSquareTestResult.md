---
title: "ChiSquareTestResult<T>"
description: "Represents the results of a Chi-Square statistical test, which is used to determine whether there is a significant  association between two categorical variables."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the results of a Chi-Square statistical test, which is used to determine whether there is a significant 
association between two categorical variables.

## For Beginners

The Chi-Square test helps determine if there's a meaningful relationship between two categorical variables.

For example, you might use this test to answer questions like:

- Is there a relationship between gender and preference for a product?
- Does treatment type affect recovery rates?
- Are survey responses distributed as expected?

The test works by:

1. Comparing observed frequencies (what you actually counted) with expected frequencies (what you would expect if there's no relationship)
2. Calculating a statistic that measures how different these frequencies are
3. Determining if this difference is statistically significant

This class stores all the information about the test results, helping you interpret whether
the observed differences are likely due to chance or represent a real association.

## How It Works

The Chi-Square test is a statistical hypothesis test that evaluates whether observed frequencies differ significantly 
from expected frequencies. It is commonly used to test the independence of two categorical variables or to assess 
the goodness of fit between observed data and a theoretical distribution. This class stores the results of such a 
test, including the test statistic, p-value, degrees of freedom, observed and expected frequencies, and whether 
the result is statistically significant. The class uses generic type parameter T to support different numeric types 
for the statistical values, such as float, double, or decimal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChiSquareTestResult` | Initializes a new instance of the ChiSquareTestResult class with all statistical values set to zero. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChiSquareStatistic` | Gets or sets the Chi-Square test statistic value. |
| `CriticalValue` | Gets or sets the critical value for the Chi-Square test at the specified significance level. |
| `IsSignificant` | Gets or sets a value indicating whether the Chi-Square test result is statistically significant. |
| `LeftExpected` | Gets or sets the expected frequencies for the left variable or category. |
| `LeftObserved` | Gets or sets the observed frequencies for the left variable or category. |
| `PValue` | Gets or sets the p-value associated with the Chi-Square test. |
| `RightExpected` | Gets or sets the expected frequencies for the right variable or category. |
| `RightObserved` | Gets or sets the observed frequencies for the right variable or category. |

