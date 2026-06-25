---
title: "DeepHit"
description: "DeepHit: A deep learning approach to survival analysis with competing risks."
section: "Reference"
---

_Regression Models_

DeepHit: A deep learning approach to survival analysis with competing risks.

## For Beginners

Unlike DeepSurv (which assumes factors affect risk proportionally at all times),
DeepHit learns the actual probability of an event at each specific time point. This is useful when:

- Risk factors affect survival differently at different times
- You want to predict exact probabilities at specific time horizons
- You have competing risks (multiple ways an event can happen)

Example: "What's the probability a patient experiences disease recurrence (risk 1) vs side effects (risk 2)
within 1 year, 2 years, or 5 years?"

Key concepts:

- Time bins: The time axis is divided into discrete bins (e.g., months 0-12, 12-24, 24-36...)
- PMF: Probability Mass Function - probability of event at each time bin
- CIF: Cumulative Incidence Function - probability of event by time t
- Survival: Probability of no event by time t

## How It Works

DeepHit directly learns the distribution of survival times without making the proportional
hazards assumption. It outputs the probability mass function (PMF) of event times across
discrete time bins and can handle multiple competing risks.

Reference: Lee, C. et al. (2018). "DeepHit: A Deep Learning Approach to Survival Analysis
with Competing Risks". AAAI Conference on Artificial Intelligence.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new DeepHit<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained DeepHit.");
```

