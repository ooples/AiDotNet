---
title: "S6Scan<T>"
description: "Provides reusable S6 (Selective Structured State Space Sequence) scan operations for Mamba-family architectures."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Provides reusable S6 (Selective Structured State Space Sequence) scan operations for Mamba-family architectures.

## For Beginners

This class contains the math that makes Mamba work.

Imagine reading a book and keeping notes:

- At each word (timestep), you update your notes (hidden state h)
- How much you update depends on the current word (selective mechanism)
- Your output is a summary based on your current notes

The "scan" processes the entire sequence step by step, updating state at each position.
This class provides two ways to do this:

- Sequential scan: processes one step at a time (simple, always correct)
- Parallel scan: processes multiple steps simultaneously using a prefix-sum trick (faster on GPUs)

## How It Works

S6 is the selective scan algorithm at the heart of Mamba. It implements the core SSM recurrence:

where A_bar and B_bar are discretized via the Zero-Order Hold (ZOH) method using input-dependent
delta (timestep) parameters. The "selective" aspect means that delta, B, and C are all functions
of the input, allowing the model to dynamically control information flow.

This static utility class extracts the scan operations from MambaBlock so they can be reused
by other SSM layers (Mamba2, hybrid architectures, etc.) without code duplication.

**Reference:** Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
https://arxiv.org/abs/2312.00752

## Methods

| Method | Summary |
|:-----|:--------|
| `ParallelScan(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs the S6 selective scan using a parallel associative scan (prefix-sum) algorithm. |
| `SequentialScanBackward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32,Int32,Int32)` | Performs the backward pass of the S6 selective scan using sequential processing. |
| `SequentialScanForward(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32,Int32,Int32,Tensor<>)` | Performs the forward pass of the S6 selective scan using sequential processing. |

