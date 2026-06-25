---
title: "IZkProofSystem"
description: "Defines the abstract interface for a zero-knowledge proof backend."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Verification`

Defines the abstract interface for a zero-knowledge proof backend.

## For Beginners

A ZK proof system provides the low-level cryptographic operations
for generating and verifying proofs. Different backends have different tradeoffs:

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this proof system. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Commit(Byte[])` | Commits to a value using the proof system's commitment scheme. |
| `GenerateRangeProof(Byte[],Byte[],Byte[],Byte[])` | Generates a range proof: proves value is in [0, upperBound]. |
| `VerifyOpening(Byte[],Byte[],Byte[])` | Verifies a commitment opening: checks that the value was indeed committed. |
| `VerifyRangeProof(Byte[],Byte[],Byte[])` | Verifies a range proof: checks that the committed value is in [0, upperBound]. |

