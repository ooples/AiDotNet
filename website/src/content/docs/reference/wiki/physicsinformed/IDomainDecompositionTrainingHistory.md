---
title: "IDomainDecompositionTrainingHistory<T>"
description: "Extended training history interface for domain decomposition PINN training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.PhysicsInformed.Interfaces`

Extended training history interface for domain decomposition PINN training.

## How It Works

For Beginners:
Domain decomposition divides a large problem domain into smaller subdomains.
Each subdomain has its own neural network, and interface conditions ensure
continuity between neighboring subdomains.

This interface tracks:

- Per-subdomain losses
- Interface continuity losses
- Overall convergence metrics

## Properties

| Property | Summary |
|:-----|:--------|
| `InterfaceLosses` | Gets the interface continuity losses per epoch. |
| `Losses` | Gets the total losses per epoch (combined from all subdomains). |
| `PhysicsLosses` | Gets the PDE residual losses per epoch (sum across all subdomains). |
| `SubdomainCount` | Gets the number of subdomains. |
| `SubdomainLosses` | Gets the losses per subdomain per epoch. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEpoch(,List<>,,)` | Records metrics for a training epoch. |

