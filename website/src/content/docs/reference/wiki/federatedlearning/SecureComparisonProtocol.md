---
title: "SecureComparisonProtocol<T>"
description: "Implements secure greater-than comparison on secret-shared values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Implements secure greater-than comparison on secret-shared values.

## For Beginners

In federated learning, operations like gradient clipping
and top-k sparsification require comparing values (e.g., "is this gradient's norm greater
than the threshold?"). But the actual gradient values are secret — no single party should
see them. Secure comparison lets parties answer "is A > B?" without revealing A or B.

## How It Works

**How it works:**

**Applications in FL:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SecureComparisonProtocol(ISecureComputationProtocol<>,Int32)` | Initializes a new instance of `SecureComparisonProtocol`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Compare(Tensor<>[],Tensor<>[])` | Compares two secret-shared tensors element-wise: returns shares of (A > B). |
| `SecureMax(Tensor<>[],Tensor<>[])` | Computes the element-wise maximum of two secret-shared tensors. |
| `SecureMin(Tensor<>[],Tensor<>[])` | Computes the element-wise minimum of two secret-shared tensors. |
| `SecureNormSquared(Tensor<>[])` | Computes the L2 norm squared of a secret-shared tensor: sum(x_i^2). |

