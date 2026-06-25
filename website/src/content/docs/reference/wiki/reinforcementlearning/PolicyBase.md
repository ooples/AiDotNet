---
title: "PolicyBase<T>"
description: "Abstract base class for policy implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ReinforcementLearning.Policies`

Abstract base class for policy implementations.
Provides common functionality for numeric operations, random number generation, and resource management.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolicyBase(Random)` | Initializes a new instance of the PolicyBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogProb(Vector<>,Vector<>)` | Computes the log probability of a given action in a given state. |
| `DeepCopy` |  |
| `Dispose(Boolean)` | Releases the unmanaged resources used by the policy and optionally releases the managed resources. |
| `GetNetworks` | Gets the neural networks used by this policy. |
| `GetParameters` |  |
| `Predict(Vector<>)` | Predicts an action for the given state (inference mode). |
| `Reset` | Resets any internal state (e.g., for recurrent policies, exploration noise). |
| `SelectAction(Vector<>,Boolean)` | Selects an action given the current state. |
| `SetParameters(Vector<>)` |  |
| `Train(Vector<>,Vector<>)` | Training is handled by RL algorithms, not directly on the policy. |
| `ValidateActionSize(Int32,Int32,String)` | Validates that an action vector has the expected size. |
| `ValidateState(Vector<>,String)` | Validates that a state vector is not null and has positive size. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_disposed` | Tracks whether the object has been disposed. |
| `_random` | Random number generator for stochastic policies. |

