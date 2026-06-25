---
title: "MultiFidelityTrainingHistory<T>"
description: "Training history for multi-fidelity PINN training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed`

Training history for multi-fidelity PINN training.

## How It Works

For Beginners:
This class tracks the training progress of a multi-fidelity PINN.
It extends the base TrainingHistory with additional metrics specific to
multi-fidelity learning:

- LowFidelityLosses: Error on cheap/approximate data
- HighFidelityLosses: Error on expensive/accurate data
- CorrelationLosses: How well the model captures the relationship between fidelity levels
- PhysicsLosses: PDE residual errors

Typical Training Dynamics:

1. Early training: Low-fidelity loss dominates (most data)
2. Mid training: High-fidelity becomes important (precision matters)
3. Late training: Physics loss should be low (PDE satisfied)

## Properties

| Property | Summary |
|:-----|:--------|
| `CorrelationLosses` | Gets the correlation losses per epoch. |
| `HighFidelityLosses` | Gets the high-fidelity data losses per epoch. |
| `LowFidelityLosses` | Gets the low-fidelity data losses per epoch. |
| `PhysicsLosses` | Gets the PDE residual losses per epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEpoch(,,,,)` | Records metrics for a training epoch. |

