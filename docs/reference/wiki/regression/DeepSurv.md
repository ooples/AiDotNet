---
title: "DeepSurv"
description: "DeepSurv: A deep learning approach to survival analysis using Cox proportional hazards."
section: "Reference"
---

_Regression Models_

DeepSurv: A deep learning approach to survival analysis using Cox proportional hazards.

## For Beginners

Survival analysis predicts "time until an event occurs." DeepSurv
is a neural network that learns to predict risk scores from your features:

- Higher risk score = event is likely to happen sooner
- Lower risk score = event is likely to happen later

What makes survival analysis unique is "censoring": some subjects haven't experienced
the event yet when the study ends. DeepSurv properly handles this by using the Cox
partial likelihood, which only compares subjects who are "at risk" at each event time.

Example applications:

- Medical: Predict patient survival time based on clinical features
- Business: Predict customer churn time based on usage patterns
- Engineering: Predict equipment failure time based on sensor data

Key outputs:

- Risk scores: Relative risk for each subject
- Survival curves: Probability of surviving past time t
- Hazard ratios: How much each feature affects risk

## How It Works

DeepSurv extends the classical Cox Proportional Hazards model by using a deep neural
network to model the log-risk function. It optimizes the negative partial log-likelihood
of the Cox model while learning complex non-linear relationships.

Reference: Katzman, J.L. et al. (2018). "DeepSurv: Personalized Treatment Recommender
System Using A Cox Proportional Hazards Deep Neural Network". BMC Medical Research Methodology.

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
    .ConfigureModel(new DeepSurv<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained DeepSurv.");
```

