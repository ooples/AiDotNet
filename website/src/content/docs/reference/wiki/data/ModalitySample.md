---
title: "ModalitySample<T>"
description: "Represents a single modality's data within a multimodal sample."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Multimodal`

Represents a single modality's data within a multimodal sample.

## For Beginners

Think of this as a labeled container.
It holds data (a tensor) plus a tag saying what kind of data it is (image, text, etc.).

## How It Works

Each ModalitySample wraps a tensor of data along with metadata about what modality
it represents and an optional key for identification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModalitySample(ModalityType,Tensor<>,String,IReadOnlyDictionary<String,Object>)` | Creates a new modality sample. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Data` | Gets the data tensor for this modality. |
| `ElementCount` | Gets the total number of elements in the data tensor. |
| `Key` | Gets an optional key to identify this modality within a multimodal sample. |
| `Metadata` | Gets optional metadata associated with this modality sample. |
| `Modality` | Gets the modality type of this sample. |
| `Shape` | Gets the shape of the data tensor. |

