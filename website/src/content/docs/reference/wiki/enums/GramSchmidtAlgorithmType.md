---
title: "GramSchmidtAlgorithmType"
description: "Represents different algorithm types for the Gram-Schmidt orthogonalization process."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different algorithm types for the Gram-Schmidt orthogonalization process.

## For Beginners

The Gram-Schmidt process is a method for converting a set of vectors into a set of 
orthogonal vectors (vectors that are perpendicular to each other).

Imagine you're in a room with walls that aren't at right angles to each other. The Gram-Schmidt process 
is like rearranging these walls so they're all perpendicular, making the room easier to measure and work with.

Why is this important in AI and machine learning?

1. Feature Independence: In machine learning, we often want features that are independent of each other. 

Orthogonal vectors represent completely independent features, which can improve model performance.

2. Dimensionality Reduction: Techniques like Principal Component Analysis (PCA) use orthogonalization to 

find the most important directions in your data.

3. Numerical Stability: Many algorithms work better when using orthogonal vectors because calculations 

become more stable and accurate.

4. Solving Systems of Equations: Orthogonal vectors make solving certain mathematical problems much easier.

The Gram-Schmidt process takes a set of vectors and, one by one, makes each new vector perpendicular to 
all previous vectors. This creates a new set of vectors that span the same space but are all perpendicular 
to each other.

This enum specifies which variation of the Gram-Schmidt algorithm to use, as there are different 
implementations with different numerical properties.

## Fields

| Field | Summary |
|:-----|:--------|
| `Classical` | Uses the Classical Gram-Schmidt algorithm for orthogonalization. |
| `Modified` | Uses the Modified Gram-Schmidt algorithm for orthogonalization. |

