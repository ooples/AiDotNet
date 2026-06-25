---
title: "ClusteringOptions<T>"
description: "Base configuration options for clustering algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Base configuration options for clustering algorithms.

## For Beginners

These are the settings you can adjust to control
how the clustering algorithm works.

Common options include:

- How many iterations to run
- When to stop (convergence threshold)
- Random seed for reproducibility (inherited Seed property)
- Which distance metric to use

## How It Works

This class provides common configuration options shared by most clustering algorithms.
Specific clustering implementations may extend this with algorithm-specific options.
Inherits from ModelOptions to provide the standard Seed property for reproducibility.

## Properties

| Property | Summary |
|:-----|:--------|
| `DistanceMetric` | Gets or sets the distance metric to use. |
| `MaxIterations` | Gets or sets the maximum number of iterations. |
| `NumInitializations` | Gets or sets the number of times to run with different random initializations. |
| `Tolerance` | Gets or sets the convergence tolerance. |
| `Verbose` | Gets or sets the verbosity level for logging. |

