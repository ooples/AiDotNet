---
title: "GradientNormRangeProof<T>"
description: "Generates and verifies proofs that a gradient's L2 norm is within a declared bound."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Generates and verifies proofs that a gradient's L2 norm is within a declared bound.

## For Beginners

Gradient clipping ensures no single client can have outsized
influence on the global model. But how does the server verify that the client actually
clipped its gradient? This class generates a cryptographic proof that ||g|| <= C
without revealing the actual norm value.

## How It Works

**How it works (simplified Bulletproofs approach):**

**Reference:** Bulletproofs (Bunz et al., S&P 2018) for logarithmic-size range proofs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GradientNormRangeProof(IZkProofSystem,Double)` | Initializes a new instance of `GradientNormRangeProof`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ProofSystemName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateNormProof(Tensor<>)` | Generates a proof that the gradient's L2 norm is within [0, C]. |
| `GenerateProof(Byte[],VerificationConstraint)` |  |
| `Verify(VerificationProof,VerificationConstraint)` |  |

