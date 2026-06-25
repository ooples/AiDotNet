---
title: "EfficientNet<T>"
description: "EfficientNet backbone for efficient feature extraction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Backbones`

EfficientNet backbone for efficient feature extraction.

## How It Works

Reference: Tan et al., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EfficientNet(EfficientNetVariant,Int32,IActivationFunction<>)` | Creates a new EfficientNet backbone. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Sum across stem + every MBConv block. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeepCopy` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activation` | Activation throughout the network. |

