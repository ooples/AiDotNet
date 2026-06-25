---
title: "MultimodalDataset<T>"
description: "A dataset of multimodal samples for training models that process multiple data types simultaneously."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Multimodal`

A dataset of multimodal samples for training models that process multiple data types simultaneously.

## For Beginners

Use this when your model needs to process multiple types of data
together, like image captioning (image + text) or audio-visual tasks (audio + video).

## How It Works

MultimodalDataset stores collections of multimodal samples and provides batching, shuffling,
and splitting capabilities. Each sample can contain any combination of modalities
(image, text, audio, etc.).

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the total number of samples in the dataset. |
| `Item(Int32)` | Gets a sample at the specified index (respects shuffle order). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(MultimodalSample<>)` | Adds a multimodal sample to the dataset. |
| `AddRange(IEnumerable<MultimodalSample<>>)` | Adds multiple multimodal samples to the dataset. |
| `GetBatchIndices(Int32)` | Iterates over the dataset in batches. |
| `GetLabelBatch(Int32,Int32)` | Gets a batch of label tensors. |
| `GetModalityBatch(Int32,Int32,String)` | Gets a batch of modality tensors by key, stacked along a new batch dimension. |
| `GetPresentKeys` | Gets all unique modality keys present across all samples. |
| `GetPresentModalities` | Gets all unique modality types present across all samples. |
| `Shuffle(Nullable<Int32>)` | Shuffles the dataset using the specified random seed. |
| `Split(Double,Double,Nullable<Int32>)` | Splits the dataset into train, validation, and test sets. |

