---
title: "PaLME<T>"
description: "PaLM-E: 562B embodied multimodal language model for robotic planning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

PaLM-E: 562B embodied multimodal language model for robotic planning.

## For Beginners

PaLM-E is a massive embodied vision-language model from Google
for robotic planning and multimodal reasoning. Default values follow the original paper
settings.

## How It Works

PaLM-E (Google, 2023) is a 562 billion parameter embodied multimodal language model that
integrates vision, language, and robot control. It injects continuous sensor observations
(images, point clouds, robot state) as tokens into the PaLM language model, enabling
embodied reasoning, task planning, and real-world robotic manipulation from natural language.

**References:**

- Paper: "PaLM-E: An Embodied Multimodal Language Model (Google, 2023)"

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsurePatchEmbedForParameterVector(Int32)` | Lazily creates _patchEmbed when the incoming parameter vector is longer than the layer-sum, indicating the saved model was trained in vision mode. |
| `ForwardForTraining(Tensor<>)` | Override the tape-driven training-mode forward to inject the same patch embedding + NCHW→BSC reshape Predict applies. |
| `GenerateFromImage(Tensor<>,String)` | Generates from image using PaLM-E's embodied multimodal approach. |
| `GetExtraTrainableLayers` | Surfaces _patchEmbed (which lives outside Layers) to the base weight-registry walker so its trainable tensors land in the streaming pool when ConfigureWeightLifetime is called. |
| `GetParameters` |  |
| `PredictAction(Tensor<>,String)` | Predicts action using PaLM-E's embodied multimodal approach. |
| `SetParameters(Vector<>)` |  |

