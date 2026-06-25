---
title: "CausalDiscoveryResult<T>"
description: "Contains the results of a causal discovery analysis, including the learned graph and convergence metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery`

Contains the results of a causal discovery analysis, including the learned graph and convergence metrics.

## For Beginners

After running causal discovery, this object tells you:

- The causal graph itself (which variables cause which others)
- Which algorithm was used
- How well the algorithm converged (did it find a good solution?)
- How sparse the graph is (fewer edges = more interpretable)

## How It Works

This result object is populated when `ConfigureCausalDiscovery()` is used on the
`AiModelBuilder`. It contains the discovered causal graph along with algorithm-specific
metrics about the optimization process.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CausalDiscoveryResult(CausalGraph<>,CausalDiscoveryAlgorithmType,CausalDiscoveryCategory,Int32,Double,Double,Boolean,TimeSpan,Dictionary<String,Double>)` | Initializes a new CausalDiscoveryResult with full convergence metrics. |
| `CausalDiscoveryResult(CausalGraph<>,CausalDiscoveryAlgorithmType,CausalDiscoveryCategory,TimeSpan)` | Initializes a new CausalDiscoveryResult from a graph and elapsed time only. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AcyclicityConstraint` | Gets the final acyclicity constraint value h(W). |
| `AdditionalMetrics` | Gets optional additional metrics specific to the algorithm used. |
| `AlgorithmUsed` | Gets the algorithm that was used for discovery. |
| `Category` | Gets the category of the algorithm used. |
| `Converged` | Gets whether the algorithm converged successfully. |
| `EdgeCount` | Gets the number of edges in the discovered graph. |
| `ElapsedTime` | Gets the wall-clock time the algorithm took to run. |
| `FinalLoss` | Gets the final loss/objective value at convergence. |
| `Graph` | Gets the discovered causal graph (DAG). |
| `GraphDensity` | Gets the graph density (fraction of possible edges present). |
| `Iterations` | Gets the number of iterations the algorithm performed. |

