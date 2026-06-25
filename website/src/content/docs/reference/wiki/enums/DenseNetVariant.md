---
title: "DenseNetVariant"
description: "Specifies the DenseNet model variant."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the DenseNet model variant.

## For Beginners

The number in the variant name (e.g., DenseNet121) indicates the total
number of layers in the network. Higher numbers mean deeper networks with potentially better
accuracy but requiring more computation time and memory.

## How It Works

DenseNet (Densely Connected Convolutional Networks) variants differ in their depth and
computational requirements. Each variant has different numbers of layers per dense block.

## Fields

| Field | Summary |
|:-----|:--------|
| `Custom` | Custom DenseNet variant for testing with minimal layers. |
| `DenseNet121` | DenseNet-121: [6, 12, 24, 16] layers per block (8M parameters). |
| `DenseNet169` | DenseNet-169: [6, 12, 32, 32] layers per block (14M parameters). |
| `DenseNet201` | DenseNet-201: [6, 12, 48, 32] layers per block (20M parameters). |
| `DenseNet264` | DenseNet-264: [6, 12, 64, 48] layers per block (33M parameters). |

