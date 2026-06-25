---
title: "EfficientNetConfiguration"
description: "Configuration options for EfficientNet neural network architectures."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for EfficientNet neural network architectures.

## For Beginners

EfficientNet is designed to achieve better accuracy with fewer
parameters by systematically scaling all network dimensions. Choose a variant based
on your accuracy requirements and computational budget.

## How It Works

EfficientNet uses compound scaling to balance network depth, width, and resolution.
Each variant (B0-B7) represents a different scale factor.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EfficientNetConfiguration(EfficientNetVariant,Int32,Int32,Nullable<Int32>,Nullable<Double>,Nullable<Double>)` | Initializes a new instance of the `EfficientNetConfiguration` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomDepthMultiplier` | Gets the custom depth multiplier (only used when Variant is Custom). |
| `CustomInputHeight` | Gets the custom input height (only used when Variant is Custom). |
| `CustomWidthMultiplier` | Gets the custom width multiplier (only used when Variant is Custom). |
| `InputChannels` | Gets the number of input channels. |
| `InputShape` | Gets the computed input shape as [channels, height, width]. |
| `NumClasses` | Gets the number of output classes for classification. |
| `Variant` | Gets the EfficientNet variant to use. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateB0(Int32)` | Creates an EfficientNet-B0 configuration (recommended default). |
| `CreateForTesting(Int32)` | Creates a minimal EfficientNet configuration optimized for fast test execution. |
| `GetDepthMultiplier` | Gets the depth multiplier for this variant. |
| `GetDropoutRate` | Gets the dropout rate for this variant. |
| `GetInputHeight` | Gets the recommended input height for this variant. |
| `GetInputWidth` | Gets the recommended input width for this variant. |
| `GetWidthMultiplier` | Gets the width multiplier for this variant. |

