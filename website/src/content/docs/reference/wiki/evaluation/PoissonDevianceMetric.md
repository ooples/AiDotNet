---
title: "PoissonDevianceMetric<T>"
description: "Computes Poisson Deviance for count data regression."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Regression`

Computes Poisson Deviance for count data regression.

## For Beginners

Poisson Deviance is designed for count data:

- Perfect for predicting counts (visitors, purchases, events)
- Handles the discrete nature of count data
- Penalizes relative errors rather than absolute errors
- Special case of Tweedie deviance with power = 1

## How It Works

Poisson Deviance = 2 × Σ(y × log(y/ŷ) - (y - ŷ)) for y > 0

**When to use:**

- Predicting number of events (clicks, purchases, calls)
- When predictions should always be positive
- When variance increases with the mean

