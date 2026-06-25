---
title: "GaussianMixtureModel<T>"
description: "Gaussian Mixture Model clustering using Expectation-Maximization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Clustering.Probabilistic`

Gaussian Mixture Model clustering using Expectation-Maximization.

## For Beginners

GMM is like soft K-Means.

Instead of saying "this point belongs to cluster 2", GMM says
"this point has 70% chance of cluster 2, 25% chance of cluster 1, 5% chance of cluster 3".

This is useful when:

- Clusters overlap (points could belong to multiple groups)
- Clusters have different shapes (some wide, some narrow)
- You need uncertainty estimates for cluster assignments

The EM algorithm works by:

- E-step: For each point, estimate probability of belonging to each cluster
- M-step: Update cluster parameters based on these probabilities
- Repeat until stable

## How It Works

GMM represents data as a mixture of Gaussian distributions. Each cluster is
characterized by its mean, covariance, and mixing weight. The EM algorithm
iteratively refines these parameters.

Algorithm steps (EM):

1. Initialize means, covariances, and weights
2. E-step: Compute responsibility of each component for each point
3. M-step: Update parameters to maximize expected log-likelihood
4. Repeat until convergence

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Clustering.Probabilistic;
using AiDotNet.Tensors.LinearAlgebra;

var data = new Matrix<double>(6, 2);
double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                    new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new GaussianMixtureModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrix(data))
    .BuildAsync();

var labels = result.Predict(data);
Console.WriteLine($"GaussianMixtureModel: clustered {labels.Length} points.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianMixtureModel(GMMOptions<>)` | Initializes a new GaussianMixtureModel instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Covariances` | Gets the component covariances. |
| `LowerBound` | Gets the lower bound (ELBO) from the last training. |
| `Means` | Gets the component means. |
| `Weights` | Gets the mixture weights. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateNewInstance` |  |
| `DeepCopy` |  |
| `FitPredict(Matrix<>)` |  |
| `GetOptions` |  |
| `Predict(Matrix<>)` |  |
| `PredictProba(Matrix<>)` | Predicts probability of each component for each sample. |
| `Sample(Int32)` | Samples from the fitted mixture model. |
| `Score(Matrix<>)` | Computes log-likelihood of data under the model. |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

