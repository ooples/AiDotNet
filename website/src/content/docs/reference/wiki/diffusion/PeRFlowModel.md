---
title: "PeRFlowModel<T>"
description: "PeRFlow (Piecewise Rectified Flow) model for few-step generation via flow straightening."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

PeRFlow (Piecewise Rectified Flow) model for few-step generation via flow straightening.

## For Beginners

Instead of trying to straighten the entire winding road from
noise to image at once, PeRFlow divides it into segments and straightens each piece.
This is easier and more effective — like fixing a winding highway by straightening
each curve individually rather than rebuilding the whole road.

## How It Works

PeRFlow divides the time interval [0,1] into K segments and straightens the flow within
each segment independently. This piecewise approach is more effective than global
straightening (Reflow) because each segment needs less straightening, achieving better
quality at 4-8 steps.

Reference: Yan et al., "PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PeRFlowModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Nullable<Int32>)` | Initializes a new PeRFlow model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `NumSegments` | Gets the number of piecewise segments. |
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

