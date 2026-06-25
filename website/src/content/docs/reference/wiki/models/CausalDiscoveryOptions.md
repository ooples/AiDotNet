---
title: "CausalDiscoveryOptions"
description: "Configuration options for causal structure discovery."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for causal structure discovery.

## For Beginners

You can use these options to configure how the causal discovery
algorithm works. If you leave everything as null, sensible defaults will be used.
The most important option is `Algorithm` which determines which method is used.

Example:

## How It Works

These options control which causal discovery algorithm is used and how it behaves.
All properties are nullable with industry-standard defaults applied internally when null.

## Properties

| Property | Summary |
|:-----|:--------|
| `AcyclicityTolerance` | Convergence tolerance for the acyclicity constraint h(W). |
| `Algorithm` | Which causal discovery algorithm to use. |
| `ConcavityParameter` | Concavity parameter (gamma) for MCP/SCAD penalty functions. |
| `CorrelationThreshold` | Correlation threshold for edge inclusion in constraint-based time-series methods. |
| `DefaultKlWeight` | Default KL divergence weight for variational methods (GAE, CausalVAE). |
| `EdgeThreshold` | Edge weight threshold for pruning. |
| `FeatureNames` | Variable/feature names to label the graph nodes. |
| `HiddenUnits` | Number of hidden units in neural network layers for deep learning methods. |
| `InitScale` | Initialization scale for weight matrices or low-rank factors. |
| `InitialLogVariance` | Initial log-variance for variational parameters in GAE and similar methods. |
| `InnerIterations` | Number of inner gradient descent steps per outer augmented Lagrangian iteration. |
| `LearningRate` | Learning rate for deep learning-based causal discovery methods. |
| `LossType` | Loss type for continuous optimization methods. |
| `MaxConditioningSetSize` | Maximum conditioning set size for constraint-based methods. |
| `MaxEpochs` | Maximum number of training epochs for deep learning-based methods. |
| `MaxIterations` | Maximum number of outer iterations for optimization-based methods. |
| `MaxKlWeight` | Maximum KL divergence weight after warm-up. |
| `MaxLag` | Maximum lag order for time-series causal discovery methods. |
| `MaxParents` | Maximum number of parents per node. |
| `MaxPenalty` | Maximum penalty parameter (rho_max) for augmented Lagrangian methods. |
| `MaxRank` | Maximum rank for low-rank matrix factorizations. |
| `MaxSegments` | Maximum number of segments for nonstationary time-series methods. |
| `Seed` | Random seed for reproducibility. |
| `SignificanceLevel` | Significance level for conditional independence tests. |
| `SobolevWeight` | Sobolev regularization weight for NOTEARS Sobolev. |
| `SparsityPenalty` | L1 sparsity penalty (lambda1). |
| `UseForFeatureSelection` | Whether to also use the discovered causal graph for feature selection in preprocessing. |
| `UseKlWarmUp` | Whether to use KL weight warm-up schedule. |

