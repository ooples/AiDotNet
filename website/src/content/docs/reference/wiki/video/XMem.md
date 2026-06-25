---
title: "XMem<T>"
description: "XMem: Long-Term Video Object Segmentation with Atkinson-Shiffrin memory model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Segmentation`

XMem: Long-Term Video Object Segmentation with Atkinson-Shiffrin memory model.

## For Beginners

XMem can track objects in hour-long videos without
running out of memory. It uses three types of memory:

- Sensory memory: Very recent frames (high detail, fast to forget)
- Working memory: Important recent frames (moderate detail)
- Long-term memory: Key historical frames (compressed, permanent)

Example usage (native mode for training):

Example usage (ONNX mode for inference):

## How It Works

XMem is designed for tracking objects in very long videos using a three-tier
memory system inspired by human memory.

**Reference:** "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model"
https://arxiv.org/abs/2207.07115

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XMem` | Creates an XMem model using native layers for training and inference. |
| `XMem(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,XMemOptions)` | Creates an XMem model using a pretrained ONNX model for inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearAllMemory` | Clears all memory banks. |
| `GetMemoryStats` | Gets memory statistics. |
| `GetOptions` |  |
| `SegmentFrame(Tensor<>)` | Segments a single frame using the memory hierarchy. |
| `TrackObjectLongTerm(List<Tensor<>>,Tensor<>)` | Tracks an object through a long video sequence. |

