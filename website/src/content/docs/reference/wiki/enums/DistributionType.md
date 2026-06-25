---
title: "DistributionType"
description: "Represents different probability distributions used in statistical modeling and machine learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different probability distributions used in statistical modeling and machine learning.

## For Beginners

Probability distributions are mathematical functions that describe how likely different outcomes are.

Think of a probability distribution like a recipe for how values are spread out:

- Some distributions create values clustered around a central point (like Normal)
- Others spread values out differently (some have long tails, some are skewed)
- Each has specific mathematical properties that make it useful for different situations

In machine learning, we use these distributions to:

- Model uncertainty and randomness
- Generate synthetic data
- Make predictions with confidence intervals
- Understand the underlying patterns in our data

Choosing the right distribution depends on what kind of data you're working with and
what assumptions you can reasonably make about how that data is generated.

## Fields

| Field | Summary |
|:-----|:--------|
| `Exponential` | A distribution that models the time between events in a process where events occur continuously and independently. |
| `Laplace` | A distribution with a sharper peak and heavier tails than the Normal distribution. |
| `LogNormal` | A distribution whose logarithm follows a Normal distribution, resulting in a skewed shape. |
| `Normal` | The bell-shaped distribution that is symmetric around its mean (also known as Gaussian distribution). |
| `Student` | A family of distributions that resemble the Normal distribution but have heavier tails (also known as t-distribution). |

