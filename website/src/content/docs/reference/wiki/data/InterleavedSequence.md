---
title: "InterleavedSequence<T>"
description: "A single interleaved sequence containing ordered segments of different modalities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Multimodal`

A single interleaved sequence containing ordered segments of different modalities.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassIndex` | Gets or sets an optional class index for this sequence. |
| `Item(Int32)` | Gets a segment at the specified position. |
| `Label` | Gets or sets an optional label tensor for this sequence. |
| `SegmentCount` | Gets the number of segments in this sequence. |
| `Segments` | Gets the segments in this sequence. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddSegment(ModalityType,Tensor<>,String)` | Adds a segment to the end of the sequence. |
| `GetByModality(ModalityType)` | Gets all segments of a specific modality type. |
| `GetModalitySequence` | Gets all segment modality types in order. |

