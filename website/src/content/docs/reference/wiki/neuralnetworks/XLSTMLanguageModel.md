---
title: "XLSTMLanguageModel<T>"
description: "Implements a full xLSTM language model: token embedding + N ExtendedLSTMLayer blocks + RMS normalization + LM head."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a full xLSTM language model: token embedding + N ExtendedLSTMLayer blocks + RMS normalization + LM head.

## For Beginners

xLSTM is a modern version of the classic LSTM that uses stronger gates
and richer memory to achieve quality competitive with Transformers and Mamba while maintaining
linear-time inference.

## How It Works

xLSTM (Extended LSTM) modernizes the classic LSTM architecture with exponential gating, new memory
structures, and residual block stacking to achieve competitive language modeling performance.

**Reference:** Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModelDimension` | Gets the model dimension (d_model). |
| `NumLayers` | Gets the number of xLSTM blocks. |
| `SupportsTraining` |  |
| `VocabSize` | Gets the vocabulary size. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` |  |

