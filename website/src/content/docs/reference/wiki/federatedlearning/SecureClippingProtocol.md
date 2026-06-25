---
title: "SecureClippingProtocol<T>"
description: "Implements secure gradient clipping without revealing gradient norms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.MPC`

Implements secure gradient clipping without revealing gradient norms.

## For Beginners

Gradient clipping limits how large a gradient can be during training.
This prevents exploding gradients and is essential for differential privacy (where gradients
must be bounded before noise is added). But in FL, we don't want the server to see the actual
gradient norms — that would leak information about clients' data.

## How It Works

**How secure clipping works:**

**Modes:**

**Reference:** SMPAI (JP Morgan, 2025) — production MPC for FL in financial services.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SecureClippingProtocol(ISecureComputationProtocol<>,Double)` | Initializes a new instance of `SecureClippingProtocol`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClipByNorm(Tensor<>[])` | Clips a secret-shared gradient vector by its L2 norm. |
| `ClipByValue(Tensor<>[])` | Clips each element of a secret-shared tensor independently to the range [-C, C]. |
| `ClipMultipleClients(IReadOnlyList<Tensor<>[]>)` | Clips multiple gradient vectors from different clients, keeping individual norms secret. |

