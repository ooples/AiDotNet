---
title: "DistanceMetricType"
description: "Represents different methods for measuring the distance or similarity between data points."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different methods for measuring the distance or similarity between data points.

## For Beginners

Distance metrics are ways to measure how similar or different two data points are.

Think of distance metrics like different ways to measure the distance between two cities:

- As the crow flies (straight line)
- By following streets (Manhattan)
- By considering terrain and obstacles (more complex metrics)

In machine learning, we use these metrics to:

- Group similar items together (clustering)
- Find nearest neighbors
- Measure how well our model is performing

Different distance metrics work better for different types of data and problems.
For example, Euclidean distance works well for continuous numerical data,
while Jaccard distance is better for comparing sets.

## Fields

| Field | Summary |
|:-----|:--------|
| `Cosine` | Measures the cosine of the angle between two vectors, focusing on orientation rather than magnitude. |
| `Euclidean` | The straight-line distance between two points in Euclidean space (also known as L2 distance). |
| `Hamming` | Counts the number of positions at which corresponding elements differ (used for strings or binary vectors). |
| `Jaccard` | Measures dissimilarity between sets by comparing elements they share versus elements they don't. |
| `Mahalanobis` | Measures distance while accounting for correlations between variables and their relative importance. |
| `Manhattan` | The sum of absolute differences between coordinates (also known as L1 distance or taxicab distance). |

