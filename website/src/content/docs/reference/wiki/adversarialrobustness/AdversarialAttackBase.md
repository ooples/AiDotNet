---
title: "AdversarialAttackBase<T, TInput, TOutput>"
description: "Base class for adversarial attack implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.AdversarialRobustness.Attacks`

Base class for adversarial attack implementations.

## For Beginners

This provides AI safety functionality. Default values follow the original paper settings.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialAttackBase(AdversarialAttackOptions<>)` | Initializes a new instance of the adversarial attack. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vectorized operations. |
| `Options` | Configuration options for the attack. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePerturbation(,)` |  |
| `ComputeL2Norm(Vector<>)` | Computes the L2 norm of a vector using vectorized operations. |
| `ComputeLInfinityNorm(Vector<>)` | Computes the L-infinity norm of a vector (maximum absolute value). |
| `Deserialize(Byte[])` |  |
| `GenerateAdversarialBatch([],[],IFullModel<,,>)` |  |
| `GenerateAdversarialExample(,,IFullModel<,,>)` |  |
| `GetDynamicShapeInfo` |  |
| `GetInputShape` | Returns the input shape for this attack configuration. |
| `GetOptions` |  |
| `GetOutputShape` | Returns the output shape for this attack configuration. |
| `LoadModel(String)` |  |
| `ProjectL2(Vector<>,)` | Projects perturbation to satisfy L2 constraint using vectorized operations. |
| `ProjectLInfinity(Vector<>,)` | Projects perturbation to satisfy L-infinity constraint using vectorized operations. |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `SignVector(Vector<>)` | Returns the sign of each element in a vector (-1, 0, or 1) using vectorized operations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `Random` | Random number generator for stochastic operations. |

