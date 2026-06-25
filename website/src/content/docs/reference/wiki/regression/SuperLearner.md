---
title: "SuperLearner"
description: "Super Learner (Stacking) ensemble for optimal model combination."
section: "Reference"
---

_Regression Models_

Super Learner (Stacking) ensemble for optimal model combination.

## For Beginners

Super Learner is an ensemble technique that:

1. Takes multiple different models (your "library" of algorithms)
2. Uses cross-validation to see how well each model predicts
3. Learns the best way to combine their predictions
4. Creates a final model that's at least as good as the best individual model

**Key advantage:** You don't have to choose which model is best - Super Learner
figures that out automatically and combines them optimally.

**Example usage:**

- Add a linear model (handles linear relationships)
- Add a random forest (handles interactions)
- Add a neural network (handles complex patterns)
- Super Learner learns to use each when appropriate

## How It Works

Super Learner combines multiple base models using cross-validated predictions to train
an optimal meta-learner. It's proven to perform at least as well as the best single
base learner (oracle inequality).

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
    .ConfigureModel(new SuperLearner<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained SuperLearner.");
```

