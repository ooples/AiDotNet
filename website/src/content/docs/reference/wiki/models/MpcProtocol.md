---
title: "MpcProtocol"
description: "Specifies the multi-party computation protocol to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the multi-party computation protocol to use.

## For Beginners

MPC lets multiple parties compute a function together without any
party revealing its private input. Different protocols trade off speed for generality:

## Fields

| Field | Summary |
|:-----|:--------|
| `AdditiveSecretSharing` | Additive secret sharing — fast for linear operations (add, scalar multiply). |
| `GarbledCircuits` | Yao's garbled circuits — supports arbitrary boolean/arithmetic computations. |
| `Hybrid` | Hybrid: additive SS for linear ops + garbled circuits for non-linear ops (compare, clip). |
| `ShamirSecretSharing` | Shamir secret sharing — threshold-based, tolerates dropouts. |

