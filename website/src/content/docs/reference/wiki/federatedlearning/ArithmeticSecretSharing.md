---
title: "ArithmeticSecretSharing<T>"
description: "Implements additive secret sharing over an arithmetic field for efficient linear operations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Implements additive secret sharing over an arithmetic field for efficient linear operations.

## For Beginners

Additive secret sharing splits a number into random "shares" that
add up to the original. For example, to share the number 42 among 3 parties:

## How It Works

**Operations on shares:**

**Reference:** This implements the standard SPDZ-style additive secret sharing protocol
used in production MPC systems.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ArithmeticSecretSharing(Int32,Int32,Nullable<Int32>)` | Initializes a new instance of `ArithmeticSecretSharing`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ReconstructionThreshold` |  |
| `SchemeName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Combine(Tensor<>[])` |  |
| `PreGenerateBeaverTriples(Int32[],Int32)` | Pre-generates Beaver triples for secure multiplication. |
| `Reconstruct(Tensor<>[])` |  |
| `ScalarMultiply(Tensor<>[],)` |  |
| `SecureAdd(Tensor<>[],Tensor<>[])` |  |
| `SecureCompare(Tensor<>[],Tensor<>[])` |  |
| `SecureMultiply(Tensor<>[],Tensor<>[])` |  |
| `Share(Tensor<>,Int32)` |  |
| `Split(Tensor<>,Int32)` |  |

