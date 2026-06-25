---
title: "IGradientCommitment<T>"
description: "Defines the contract for committing to gradient values before revealing them."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Verification`

Defines the contract for committing to gradient values before revealing them.

## For Beginners

In FL, a malicious client could wait to see other clients' updates
and then craft its own update to manipulate the result (an "adaptive attack"). Commitment
schemes prevent this: each client first sends a "commitment" (a cryptographic lock on its
gradient), and only after all commitments are received does anyone reveal the actual values.

## How It Works

**Properties:**

**Homomorphic commitments** (like Pedersen) have an additional property: the server
can verify that the sum of committed values matches the committed sum, without seeing
individual values.

## Methods

| Method | Summary |
|:-----|:--------|
| `Commit(Tensor<>)` | Creates a commitment to a gradient tensor. |
| `Open(GradientCommitmentData<>)` | Opens a commitment, revealing the original gradient value. |
| `Verify(GradientCommitmentData<>)` | Verifies that an opened commitment matches the original commitment. |
| `VerifyAggregation(IReadOnlyList<GradientCommitmentData<>>,GradientCommitmentData<>)` | Verifies that the sum of individual commitments matches a claimed aggregate commitment. |

