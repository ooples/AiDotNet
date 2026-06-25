---
title: "ISecretSharingScheme<T>"
description: "Defines the contract for a secret sharing scheme that splits and recombines tensor values."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.MPC`

Defines the contract for a secret sharing scheme that splits and recombines tensor values.

## For Beginners

Secret sharing lets you split a secret number into multiple
"shares" that individually look random. The original secret can only be recovered when
enough shares are combined. This is the foundation of MPC.

## How It Works

**Types of secret sharing:**

**Extends:** This generic interface works with tensors and complements the existing
`ShamirSecretSharing` class (which operates on raw byte arrays for crypto-level operations).

## Properties

| Property | Summary |
|:-----|:--------|
| `ReconstructionThreshold` | Gets the minimum number of shares required to reconstruct the secret. |
| `SchemeName` | Gets the name of this secret sharing scheme. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Combine(Tensor<>[])` | Reconstructs the original tensor from a set of shares. |
| `Split(Tensor<>,Int32)` | Splits a tensor value into secret shares for the specified number of parties. |

