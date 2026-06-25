---
title: "ControlNetCondition<T>"
description: "Represents a single control condition input for multi-control ControlNet composition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

Represents a single control condition input for multi-control ControlNet composition.

## For Beginners

This is like a labeled ingredient with a measured amount. Each
control condition says "I'm a depth map with 80% influence" or "I'm a sketch with 50%
influence", so the model knows how much to pay attention to each control signal.

## How It Works

Pairs a conditioning image/tensor with its type identifier and a blending weight,
enabling weighted composition of multiple control signals. Used by ControlNet++ and
ControlNet-Union models that support simultaneous multi-condition inputs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetCondition(ControlNetConditionType,Tensor<>,Double)` | Represents a single control condition input for multi-control ControlNet composition. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConditionImage` | The conditioning tensor (preprocessed control image). |
| `ConditionType` | The semantic type of this control condition. |
| `Weight` | Blending weight for this condition (0.0 to 1.0, default 1.0). |

