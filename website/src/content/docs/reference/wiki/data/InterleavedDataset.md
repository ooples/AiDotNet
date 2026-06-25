---
title: "InterleavedDataset<T>"
description: "A dataset where each sample is an interleaved sequence of modality segments, such as alternating image-text-image-text in vision-language models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Multimodal`

A dataset where each sample is an interleaved sequence of modality segments,
such as alternating image-text-image-text in vision-language models.

## For Beginners

Think of a web page with alternating text paragraphs and images.
This dataset represents each "page" as a sequence of typed segments:

## How It Works

Interleaved sequences are common in modern multimodal models like GPT-4V, Gemini, and Flamingo,
where images and text are interspersed in a single sequence. Each segment has a modality type
and position, enabling the model to process mixed-modality input natively.

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the total number of interleaved sequences in the dataset. |
| `Item(Int32)` | Gets a sequence at the specified index. |
| `MaxSegmentCount` | Gets the maximum number of segments across all sequences. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(InterleavedSequence<>)` | Adds an interleaved sequence to the dataset. |
| `AddRange(IEnumerable<InterleavedSequence<>>)` | Adds multiple sequences to the dataset. |
| `GetSegmentsByModality(ModalityType)` | Extracts all segments of a given modality type across all sequences. |

