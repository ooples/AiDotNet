---
title: "HashCommitmentScheme<T>"
description: "Implements a hash-based commitment scheme using SHA-256."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Implements a hash-based commitment scheme using SHA-256.

## For Beginners

This is the simplest commitment scheme. To commit to a value v:

## How It Works

**Security:**

**Limitation:** Hash commitments are NOT homomorphic — the server cannot verify sums
of committed values. Use `PedersenCommitment` if you need that property.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HashCommitmentScheme(Int32)` | Initializes a new instance of `HashCommitmentScheme`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Commit(Byte[])` |  |
| `Commit(Tensor<>)` |  |
| `GenerateRangeProof(Byte[],Byte[],Byte[],Byte[])` |  |
| `Open(GradientCommitmentData<>)` |  |
| `Verify(GradientCommitmentData<>)` |  |
| `VerifyAggregation(IReadOnlyList<GradientCommitmentData<>>,GradientCommitmentData<>)` | Not supported for hash commitments (not homomorphic). |
| `VerifyOpening(Byte[],Byte[],Byte[])` |  |
| `VerifyRangeProof(Byte[],Byte[],Byte[])` |  |

