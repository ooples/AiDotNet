---
title: "InterleavedSegment<T>"
description: "A single segment within an interleaved sequence."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Multimodal`

A single segment within an interleaved sequence.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InterleavedSegment(ModalityType,Tensor<>,Int32,String)` | Creates a new interleaved segment. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Data` | Gets the data tensor for this segment. |
| `Key` | Gets an optional key for this segment. |
| `Modality` | Gets the modality type of this segment. |
| `Position` | Gets the position of this segment within its parent sequence (0-based). |

