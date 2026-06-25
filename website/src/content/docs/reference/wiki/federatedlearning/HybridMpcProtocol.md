---
title: "HybridMpcProtocol<T>"
description: "Combines arithmetic secret sharing (for linear operations) with garbled circuits (for non-linear operations) into a single hybrid MPC protocol."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Combines arithmetic secret sharing (for linear operations) with garbled circuits
(for non-linear operations) into a single hybrid MPC protocol.

## For Beginners

Different MPC techniques are efficient for different operations:

## How It Works

The hybrid approach uses arithmetic sharing by default and automatically switches to
garbled circuits when a non-linear operation is needed. Share conversion between the two
representations uses oblivious transfer.

**In FL, the typical workflow is:**

**Reference:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridMpcProtocol(MpcOptions)` | Initializes a new instance of `HybridMpcProtocol`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ArithmeticScheme` | Gets the arithmetic secret sharing scheme used internally. |
| `BooleanScheme` | Gets the boolean secret sharing scheme used internally. |
| `CircuitEvaluator` | Gets the garbled circuit evaluator. |
| `CircuitGenerator` | Gets the garbled circuit generator used for non-linear operations. |
| `Clipping` | Gets the secure clipping protocol. |
| `Comparison` | Gets the secure comparison protocol. |
| `ObliviousTransfer` | Gets the oblivious transfer protocol. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Reconstruct(Tensor<>[])` |  |
| `ScalarMultiply(Tensor<>[],)` |  |
| `SecureAdd(Tensor<>[],Tensor<>[])` |  |
| `SecureClipByValue(Tensor<>[])` | Performs secure element-wise clipping on secret-shared gradients. |
| `SecureClipGradient(Tensor<>[])` | Performs secure gradient clipping on secret-shared gradients. |
| `SecureClippedAggregation(IReadOnlyList<Tensor<>[]>,IReadOnlyList<Double>)` | Performs secure aggregation with clipping: clips each client's gradient, then aggregates. |
| `SecureCompare(Tensor<>[],Tensor<>[])` |  |
| `SecureMultiply(Tensor<>[],Tensor<>[])` |  |
| `SecureWeightedSum(IReadOnlyList<Tensor<>[]>,IReadOnlyList<Double>)` | Securely aggregates gradients from multiple clients using weighted sum. |
| `Share(Tensor<>,Int32)` |  |

