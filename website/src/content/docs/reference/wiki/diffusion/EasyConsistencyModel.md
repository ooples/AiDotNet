---
title: "EasyConsistencyModel<T>"
description: "Easy Consistency Tuning (ECT) model for simple, stable consistency model training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Easy Consistency Tuning (ECT) model for simple, stable consistency model training.

## For Beginners

Training consistency models from scratch is tricky — small
mistakes in hyperparameters cause training to diverge. ECT makes it easy by starting
with small noise differences (easy mappings) and gradually increasing difficulty.
Think of it like learning to long-jump: you start with short distances and work up.

## How It Works

ECT simplifies consistency model training by using a progressive training curriculum
that gradually increases the difficulty of the consistency mapping. This removes the
need for careful hyperparameter tuning and produces stable training from any pretrained
diffusion model.

Reference: Geng et al., "Consistency Models Made Easy", NeurIPS 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EasyConsistencyModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Nullable<Int32>)` | Initializes a new Easy Consistency Tuning model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

