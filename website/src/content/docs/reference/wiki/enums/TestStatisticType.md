---
title: "TestStatisticType"
description: "Represents different types of statistical tests used to evaluate hypotheses and determine significance in data analysis."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different types of statistical tests used to evaluate hypotheses and determine significance in data analysis.

## How It Works

**For Beginners:** Statistical tests help us decide if patterns we see in data are real or just due to chance.

Think of statistical tests like different tools in a toolbox - each one is designed for specific situations:

Imagine you're trying to determine if a coin is fair (50% chance of heads). You could flip it 100 times
and count how many heads you get. If you get exactly 50 heads, it seems fair. But what if you get 55 heads?
Or 60? At what point do you decide the coin is unfair? Statistical tests give us mathematical ways to
make these decisions based on probability rather than just guessing.

Different tests are designed for different types of data and questions, just like you'd use different
tools for different home repair jobs.

These tests calculate a "p-value" - the probability that the pattern you observed could happen by random chance.
A small p-value (typically < 0.05) suggests the pattern is statistically significant and not just random.

## Fields

| Field | Summary |
|:-----|:--------|
| `ChiSquare` | A statistical test used to determine if there is a significant association between categorical variables. |
| `FTest` | A statistical test that compares the variances of two or more groups to determine if they are significantly different. |
| `MannWhitneyU` | A non-parametric test that compares two independent samples without assuming they follow a normal distribution. |
| `PermutationTest` | A resampling-based test that repeatedly shuffles observed data to determine if patterns are statistically significant. |
| `TTest` | A statistical test used to determine if there is a significant difference between the means of two groups. |

