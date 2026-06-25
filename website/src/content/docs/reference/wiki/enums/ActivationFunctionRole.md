---
title: "ActivationFunctionRole"
description: "Defines the functional roles of activation functions in neural networks."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the functional roles of activation functions in neural networks.

## For Beginners

This enum helps organize the different "jobs" that activation functions perform
in neural networks, similar to how different workers in a factory have different specialized roles.

## How It Works

Different parts of neural networks typically require different activation behaviors.
This enum categorizes activation functions by their role rather than by their mathematical form.

## Fields

| Field | Summary |
|:-----|:--------|
| `Attention` | Used for attention mechanisms. |
| `Cell` | Used for memory cell state updates. |
| `Gate` | Used for gate mechanisms that control information flow. |
| `Hidden` | Used for standard hidden layer activations. |
| `Normalization` | Used for normalization functions. |
| `Output` | Used for output layer activations. |
| `Probability` | Used for probability distributions. |

