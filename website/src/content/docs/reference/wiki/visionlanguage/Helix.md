---
title: "Helix<T>"
description: "Helix: dual-system vision-language-action model for full-body humanoid control (Figure AI, 2025, arXiv:2502.07092)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

Helix: dual-system vision-language-action model for full-body humanoid control
(Figure AI, 2025, arXiv:2502.07092).

## For Beginners

Helix is the first robot brain that's good enough to drive a real
humanoid through dexterous tasks like loading a dishwasher. The trick is two networks: a smart-but-
slow one decides intent ("pick up the red mug") and a fast-but-narrow one figures out the exact
motor commands needed to do it 200 times per second.

## How It Works

Helix is the first generalist VLA model deployed on a real humanoid for whole upper-body
dexterous manipulation. Its defining architectural choice is the **dual-system, dual-rate**
split between a slow VLM reasoner and a fast visuomotor policy:

**Paper-faithful pieces implemented here:**

**What is NOT verified in-session:**

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `System1ToSystem2Ratio` | Default ratio of S1 to S2 invocations (paper §4.1: 200 Hz S1, 7–9 Hz S2 → ~22:1). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDualSystemRunner` | Creates a `HelixDualSystemRunner` wired to this model's S1/S2 callbacks for streaming-control use. |
| `EmbedInstructionTokens(Tensor<>)` | Looks up instruction-token embeddings through the learned `_tokenEmbedding` table (Figure AI 2025, §3.2). |
| `GetParameters` |  |
| `PredictAction(Tensor<>,String)` | Predicts `ActionDimension` × `PredictionHorizon` continuous joint commands using Helix's dual-system inference path: one S2 invocation produces the semantic latent, then S1 rolls out `PredictionHorizon` joint-command tensors conditioned on… |
| `SetParameters(Vector<>)` |  |
| `System1Forward(Tensor<>,Tensor<>)` | **System 1** forward pass: runs the fast 80M visuomotor policy at every control tick (200 Hz target). |
| `System2Forward(Tensor<>,String)` | **System 2** forward pass: runs the full VLM (vision encoder + language decoder) and returns the semantic latent that conditions the fast System-1 controller. |
| `TanhClampJointCommands(Tensor<>,Int32)` | Squashes the action-head output into a per-joint tanh-bounded velocity command. |

