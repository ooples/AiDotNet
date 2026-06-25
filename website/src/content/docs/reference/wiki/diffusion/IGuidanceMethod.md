---
title: "IGuidanceMethod<T>"
description: "Interface for guidance methods that modify noise predictions during diffusion sampling."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Diffusion.Guidance`

Interface for guidance methods that modify noise predictions during diffusion sampling.

## For Beginners

Guidance is like an invisible hand that steers the AI toward
your prompt. Standard CFG compares "with prompt" vs "without prompt" predictions.
Advanced methods like PAG and SAG use attention manipulation for better results.

## How It Works

Guidance methods control how the diffusion model balances prompt adherence and image
quality during generation. Different methods offer different trade-offs.

## Properties

| Property | Summary |
|:-----|:--------|
| `GuidanceType` | Gets the type of guidance this method implements. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(Tensor<>,Tensor<>,Double,Double)` | Applies guidance to combine conditional and unconditional noise predictions. |

