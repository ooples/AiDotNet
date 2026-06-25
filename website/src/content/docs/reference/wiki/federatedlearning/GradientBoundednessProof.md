---
title: "GradientBoundednessProof<T>"
description: "Proves that each gradient component is within [-B, B] (element-wise range proof)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Proves that each gradient component is within [-B, B] (element-wise range proof).

## For Beginners

While norm-bound proofs constrain the overall magnitude of a gradient,
element-wise bound proofs ensure no single gradient component is abnormally large. This is
important because a gradient with a small norm could still have one extreme component
(a "needle" attack).

## How It Works

**How it works:**

**Optimization:** Rather than proving each element individually (expensive), we use
a batched proof that commits to all elements at once and proves the bound in aggregate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientBoundednessProof(IZkProofSystem,Double)` | Initializes a new instance of `GradientBoundednessProof`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ProofSystemName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateBoundednessProof(Tensor<>)` | Generates a proof that all gradient elements are within [-B, B]. |
| `GenerateProof(Byte[],VerificationConstraint)` |  |
| `Verify(VerificationProof,VerificationConstraint)` |  |

