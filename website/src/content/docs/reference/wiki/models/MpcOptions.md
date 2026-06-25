---
title: "MpcOptions"
description: "Configuration options for multi-party computation protocols in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for multi-party computation protocols in federated learning.

## For Beginners

These settings control how MPC operates within the FL pipeline.
The defaults use additive secret sharing with semi-honest security, which is the fastest
configuration suitable for environments where participants are trusted to follow the protocol.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseObliviousTransferCount` | Gets or sets the number of base OTs to perform for OT extension. |
| `ClippingNormThreshold` | Gets or sets the gradient clipping norm threshold for secure clipping. |
| `CovertDeterrenceFactor` | Gets or sets the covert security deterrence factor (probability of detecting cheating). |
| `EnableFreeXor` | Gets or sets whether to enable free XOR optimization in garbled circuits. |
| `EnableHalfGates` | Gets or sets whether to enable half-gates optimization in garbled circuits. |
| `FieldBitLength` | Gets or sets the prime field modulus bit length for arithmetic secret sharing. |
| `ObliviousTransferBatchSize` | Gets or sets the batch size for OT extension operations. |
| `Protocol` | Gets or sets the MPC protocol to use. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SecurityModel` | Gets or sets the adversary model. |
| `SecurityParameterBits` | Gets or sets the security parameter in bits (e.g., 128, 256). |
| `Threshold` | Gets or sets the reconstruction threshold for Shamir-based protocols. |

