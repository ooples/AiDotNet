---
title: "ToonCrafter<T>"
description: "ToonCrafter generative cartoon interpolation for large non-linear animated motion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

ToonCrafter generative cartoon interpolation for large non-linear animated motion.

## For Beginners

ToonCrafter is designed specifically for cartoons and animations.
Unlike real-world video where objects move smoothly, cartoon characters can teleport,
stretch, and move in impossible ways. This model understands these animation-specific
motion patterns and generates in-between frames that look natural for animated content.

## How It Works

**References:**

- Paper: "ToonCrafter: Generative Cartoon Interpolation" (2024)

ToonCrafter specializes in cartoon and animation frame interpolation where motions are
large, non-linear, and don't follow physical constraints. It uses a latent diffusion
approach adapted for animation:

- Dual-reference conditioning: conditions the diffusion process on both start and end

cartoon frames simultaneously, using cross-attention to attend to features from both

- Sketch-guided generation: optionally uses extracted sketch/edge maps as structural

guidance, ensuring generated frames maintain the line art style and character proportions

- Toon-adapted noise schedule: a modified diffusion noise schedule that works better with

the flat colors and sharp edges typical of cartoon/animation content

- Large motion capability: handles the extreme, physically-unrealistic motions common in

animation (e.g., squash-and-stretch, sudden direction changes, exaggerated physics)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ToonCrafter(NeuralNetworkArchitecture<>,String,ToonCrafterOptions)` | Creates a ToonCrafter model for ONNX inference. |
| `ToonCrafter(NeuralNetworkArchitecture<>,ToonCrafterOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a ToonCrafter model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

