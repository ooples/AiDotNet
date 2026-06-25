---
title: "PsiBase"
description: "Base class providing shared functionality for PSI protocol implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.FederatedLearning.PSI`

Base class providing shared functionality for PSI protocol implementations.

## For Beginners

This base class handles common tasks like input validation,
building alignment mappings from intersection results, and timing execution.
Each concrete PSI protocol only needs to implement the core intersection algorithm.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProtocolName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildAlignmentResult(IReadOnlyList<String>,IReadOnlyList<String>,IReadOnlyList<String>)` | Builds alignment mappings from a set of intersection IDs and the original ID lists. |
| `ComputeCardinality(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `ComputeExactCardinality(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Computes the exact cardinality using the protocol-specific algorithm. |
| `ComputeExactIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Computes the exact intersection using the protocol-specific algorithm. |
| `ComputeFuzzyIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` | Computes fuzzy intersection by delegating to the appropriate fuzzy matcher. |
| `ComputeIntersection(IReadOnlyList<String>,IReadOnlyList<String>,PsiOptions)` |  |
| `CreateFuzzyMatcher(FuzzyMatchStrategy)` | Creates a fuzzy matcher for the given strategy. |
| `NormalizeWhitespace(String)` | Normalizes whitespace in a string by collapsing runs of whitespace to single spaces and trimming. |

