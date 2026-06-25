---
title: "PedersenCommitment<T>"
description: "Implements Pedersen commitment scheme — additively homomorphic for verifiable aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Implements Pedersen commitment scheme — additively homomorphic for verifiable aggregation.

## For Beginners

Pedersen commitments are special because they're "additively
homomorphic": if you commit to values a and b separately, anyone can combine the two
commitments to get a valid commitment to a+b — without knowing a or b.

## How It Works

**How it works:**

**In FL:** The server receives commitments from all clients. It can multiply them
together to get a commitment to the sum. Then when clients open their commitments, the
server verifies both individual values and the aggregate — detecting any manipulation.

**Reference:** Pedersen (CRYPTO 1991). Used in RiseFL (VLDB 2024) for scalable FL verification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PedersenCommitment(Int32)` | Initializes a new instance of `PedersenCommitment`. |

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
| `VerifyAggregation(IReadOnlyList<GradientCommitmentData<>>,GradientCommitmentData<>)` |  |
| `VerifyOpening(Byte[],Byte[],Byte[])` |  |
| `VerifyRangeProof(Byte[],Byte[],Byte[])` |  |

