---
title: "AcquisitionFunctionType"
description: "Represents different types of acquisition functions used in Bayesian optimization."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different types of acquisition functions used in Bayesian optimization.

## For Beginners

Acquisition functions help an AI system decide where to look next when searching 
for the best solution to a problem.

Imagine you're trying to find the highest point in a mountain range that's covered in fog:

- You've already explored a few spots and know their heights
- Based on these measurements, you can make educated guesses about unexplored areas
- But you need to decide: should you explore areas that look promising based on what you know so far,

or should you check completely unexplored areas that might contain surprises?

This is the "exploration vs. exploitation" trade-off, and acquisition functions help balance it.

Acquisition functions are particularly useful when:

- Testing each possible solution is expensive or time-consuming
- You want to find the best solution with as few attempts as possible
- The relationship between inputs and outputs is complex

Common applications include hyperparameter tuning in machine learning, experimental design,
and optimizing complex systems where each test is costly.

## Fields

| Field | Summary |
|:-----|:--------|
| `ExpectedImprovement` | Expected Improvement acquisition function that focuses on areas likely to improve upon the current best solution. |
| `ProbabilityOfImprovement` | Probability of Improvement acquisition function that maximizes the probability of finding better solutions. |
| `UpperConfidenceBound` | Upper Confidence Bound acquisition function that balances exploration and exploitation. |

