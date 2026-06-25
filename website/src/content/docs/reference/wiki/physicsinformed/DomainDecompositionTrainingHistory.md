---
title: "DomainDecompositionTrainingHistory<T>"
description: "Training history for domain decomposition PINN training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed`

Training history for domain decomposition PINN training.

## How It Works

For Beginners:
This class tracks the training progress of a domain decomposition PINN.
It extends the base TrainingHistory with additional metrics specific to
domain decomposition:

- SubdomainLosses: Per-subdomain PDE and data losses
- InterfaceLosses: Continuity errors at subdomain boundaries
- PhysicsLosses: Total PDE residual across all subdomains

Key Observations During Training:

1. Subdomain losses should decrease independently
2. Interface losses ensure solution continuity
3. If one subdomain loss is much higher, it may need more capacity
4. Interface losses are critical for global solution quality

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DomainDecompositionTrainingHistory(Int32)` | Initializes a new instance with the specified number of subdomains. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InterfaceLosses` | Gets the interface continuity losses per epoch. |
| `PhysicsLosses` | Gets the PDE residual losses per epoch. |
| `SubdomainCount` | Gets the number of subdomains. |
| `SubdomainLosses` | Gets the losses per subdomain per epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEpoch(,List<>,,)` | Records metrics for a training epoch. |

