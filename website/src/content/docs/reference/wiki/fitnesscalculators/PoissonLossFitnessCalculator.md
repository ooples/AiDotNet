---
title: "PoissonLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Poisson Loss to evaluate model performance, particularly for count-based prediction tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Poisson Loss to evaluate model performance, particularly for count-based prediction tasks.

## For Beginners

This calculator helps evaluate how well your model is performing on count-based prediction tasks,
which are problems where you're predicting the number of times something happens in a fixed interval.

Poisson Loss is designed specifically for problems where:

- You're predicting counts or frequencies (whole numbers, 0 or greater)
- The events occur independently of each other
- The average rate of events is constant

Examples of count-based prediction problems:

- Number of customer arrivals per hour
- Number of goals scored in a soccer match
- Number of website visits per day
- Number of defects in a manufacturing process
- Number of calls to a call center per hour

How Poisson Loss works:

- It's based on the Poisson distribution, which models random events occurring over a fixed interval
- It's particularly good for data where most values are small counts (0, 1, 2, etc.) but occasionally have larger values
- It penalizes both overestimation and underestimation, but in a way that's appropriate for count data

Think of it like this:
Imagine you're predicting how many customers will visit a store each hour:

- Some hours might have 0 customers
- Most hours might have 5-10 customers
- Occasionally, there might be 20+ customers
- Poisson Loss is designed to handle this kind of pattern well

Key characteristics:

- It's specifically designed for count data
- It works well when the variance of your data increases with the mean
- Lower values are better (0 would be perfect predictions)
- It assumes non-negative values (counts can't be negative)

When to use this calculator:

- When your target values are counts (whole numbers, 0 or greater)
- When your data follows a Poisson-like distribution (many small values, fewer large values)
- When you're predicting the frequency of events in a fixed time or space interval

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PoissonLossFitnessCalculator(DataSetType)` | Initializes a new instance of the PoissonLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Poisson Loss fitness score for the given dataset. |

