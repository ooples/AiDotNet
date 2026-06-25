---
title: "HybridBlockScheduler<T>"
description: "Implements a composable hybrid block that schedules SSM and attention layers according to configurable patterns used in modern hybrid architectures (Jamba, Zamba, Samba)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements a composable hybrid block that schedules SSM and attention layers according to
configurable patterns used in modern hybrid architectures (Jamba, Zamba, Samba).

## For Beginners

This is like a playlist that decides which type of layer
(SSM or attention) to use at each position in the network.

Imagine building a team:

- Mamba blocks are fast workers who process one thing at a time very efficiently
- Attention blocks are thorough workers who compare everything to everything else

A hybrid schedule picks the best worker for each position:

- Jamba: mostly fast workers, with a thorough worker every Nth position
- Zamba: fast workers sharing one set of thorough worker notes (shared attention)
- Samba: alternating fast workers and thorough workers with limited vision (sliding window)

The result is faster than pure attention but more capable than pure SSM.

## How It Works

Pure SSM models (all Mamba) and pure Transformer models (all attention) each have strengths:

- SSM: O(n) linear complexity, good at long-range sequential patterns
- Attention: O(n^2) quadratic complexity, but excellent at in-context learning and recall

Hybrid architectures combine both to get the best of each. The `HybridBlockScheduler`
lets you define the mixing pattern.

Each block in the schedule applies: LayerNorm -> SubLayer -> Residual Connection.
This follows the pre-norm Transformer convention used by all modern architectures.

**References:**

- Jamba: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", 2024
- Zamba: Glorioso et al., "Zamba: A Compact 7B SSM Hybrid Model", 2024
- Samba: Ren et al., "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HybridBlockScheduler(Int32,ILayer<>[],Boolean[],HybridSchedulePattern,Int32,IActivationFunction<>)` | Creates a new hybrid block scheduler. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension. |
| `NumBlocks` | Gets the number of blocks in the schedule. |
| `ParameterCount` | Gets the total number of trainable parameters across all blocks and norms. |
| `SchedulePattern` | Gets the schedule pattern. |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateJambaSchedule(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates a Jamba-style hybrid schedule where every Nth layer is attention and the rest are SSM. |
| `CreateSambaSchedule(Int32,Int32,Int32,Int32,Int32)` | Creates a Samba-style hybrid schedule alternating Mamba blocks and sliding window attention. |
| `CreateZambaSchedule(Int32,Int32,Int32,Int32,Int32,Int32)` | Creates a Zamba-style hybrid schedule with attention layers interleaved with SSM blocks. |
| `Forward(Tensor<>)` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

