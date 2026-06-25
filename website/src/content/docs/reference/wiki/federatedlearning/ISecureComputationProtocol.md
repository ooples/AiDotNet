---
title: "ISecureComputationProtocol<T>"
description: "Defines the contract for a multi-party computation protocol that can perform secure arithmetic and comparison operations on secret-shared values."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.MPC`

Defines the contract for a multi-party computation protocol that can perform secure
arithmetic and comparison operations on secret-shared values.

## For Beginners

Imagine several parties each hold a piece of a number (a "share").
None of them know the original number, but together they can add, multiply, or compare
secret numbers by exchanging specially crafted messages — all without ever revealing the
actual values. This interface defines those operations.

## How It Works

**How it works in FL:**

## Methods

| Method | Summary |
|:-----|:--------|
| `Reconstruct(Tensor<>[])` | Reconstructs the plaintext value from a set of shares. |
| `ScalarMultiply(Tensor<>[],)` | Multiplies shares by a public (non-secret) scalar. |
| `SecureAdd(Tensor<>[],Tensor<>[])` | Performs element-wise secure addition of two sets of shares. |
| `SecureCompare(Tensor<>[],Tensor<>[])` | Performs secure comparison: is `sharesA` element-wise greater than `sharesB`? Returns shares of a binary tensor (1 where true, 0 where false). |
| `SecureMultiply(Tensor<>[],Tensor<>[])` | Performs element-wise secure multiplication of two sets of shares. |
| `Share(Tensor<>,Int32)` | Splits a plaintext value into secret shares for the specified number of parties. |

