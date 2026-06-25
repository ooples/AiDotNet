---
title: "FinchLanguageModel<T>"
description: "Implements a full RWKV-6 \"Finch\" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full RWKV-6 "Finch" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head.

## For Beginners

Finch (RWKV-6) builds on Eagle by adding the ability to
dynamically decide how much context from previous tokens to blend into the current one.
Think of it as a reader who can dynamically adjust their focus: sometimes reading word
by word, other times absorbing whole phrases. This adaptive blending helps it better
capture complex language patterns while maintaining the same memory-efficient inference
as its predecessor.

## How It Works

Finch extends Eagle (RWKV-5) with data-dependent token shifting via a LoRA-based mechanism,
allowing the model to dynamically adjust how much to blend current and previous tokens.

**Reference:** Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", 2024.
https://arxiv.org/abs/2404.05892

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of Finch blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

