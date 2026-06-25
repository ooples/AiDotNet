---
title: "IMultiFidelityTrainingHistory<T>"
description: "Extended training history interface for multi-fidelity PINN training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Extended training history interface for multi-fidelity PINN training.

## How It Works

For Beginners:
Multi-fidelity training uses data from multiple sources with different accuracy levels:

- Low-fidelity: Cheap but less accurate (e.g., coarse simulations, simplified models)
- High-fidelity: Expensive but accurate (e.g., fine simulations, experiments)

This interface tracks metrics for each fidelity level during training.

## Properties

| Property | Summary |
|:-----|:--------|
| `CorrelationLosses` | Gets the correlation losses per epoch (measures agreement between fidelity levels). |
| `HighFidelityLosses` | Gets the high-fidelity data losses per epoch. |
| `Losses` | Gets the total losses per epoch (combined from all fidelity levels). |
| `LowFidelityLosses` | Gets the low-fidelity data losses per epoch. |
| `PhysicsLosses` | Gets the PDE residual losses per epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEpoch(,,,,)` | Records metrics for a training epoch. |

