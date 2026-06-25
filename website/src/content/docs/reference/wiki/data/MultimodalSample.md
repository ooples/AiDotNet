---
title: "MultimodalSample<T>"
description: "A collection of modality samples representing one multimodal data point."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Multimodal`

A collection of modality samples representing one multimodal data point.

## For Beginners

This is like a folder containing different types of data
that all describe the same thing. For example:

## How It Works

A MultimodalSample groups multiple modalities together as a single training example.
For instance, an image-text pair for captioning would contain an Image modality
and a Text modality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultimodalSample(IEnumerable<ModalitySample<>>)` | Creates a new multimodal sample from a list of modality samples. |
| `MultimodalSample(ModalitySample<>[])` | Creates a new multimodal sample from a collection of modality samples. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassIndex` | Gets or sets an optional integer class index. |
| `Item(String)` | Gets a modality sample by its key. |
| `Keys` | Gets the available keys for accessing modalities. |
| `Label` | Gets optional label information for this sample. |
| `LabelName` | Gets or sets an optional string label (for classification). |
| `Modalities` | Gets the modality samples in this multimodal sample. |
| `ModalityCount` | Gets the number of modalities in this sample. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetByType(ModalityType)` | Gets all modality samples of a specific type. |
| `HasKey(String)` | Checks whether a modality key exists in this sample. |
| `HasModality(ModalityType)` | Checks whether this sample contains any modality of the specified type. |
| `TryGetModality(String,ModalitySample<>)` | Tries to get a modality sample by key. |

