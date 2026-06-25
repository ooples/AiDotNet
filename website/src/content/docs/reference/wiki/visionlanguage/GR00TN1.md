---
title: "GR00TN1<T>"
description: "GR00T N1: NVIDIA's open foundation model for generalist humanoid robots, combining a SigLIP + Eagle-2 vision-language System-2 reasoner with a flow-matching DiT action head as System 1 (NVIDIA 2025, arXiv:2503.14734)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

GR00T N1: NVIDIA's open foundation model for generalist humanoid robots, combining a SigLIP +
Eagle-2 vision-language System-2 reasoner with a flow-matching DiT action head as System 1
(NVIDIA 2025, arXiv:2503.14734).

## For Beginners

GR00T N1 is the first big-tech open humanoid brain. The fast policy
doesn't just predict joints directly — it predicts a noise-to-data flow, the same trick image
generators like Stable Diffusion use. This lets it produce smooth, physically-plausible joint
trajectories instead of jerky direct regressions.

## How It Works

GR00T N1 is the first publicly released foundation model trained on the GR00T-1B humanoid
dataset. Like Helix, it uses a dual-system architecture, but the System-1 policy is a
**flow-matching** DiT (Lipman et al. 2023) rather than a direct regression head. At
inference the model:

**Paper-faithful pieces implemented here:**

**What is NOT verified in-session:**

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionHead` | The flow-matching action head used by `String)`. |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDualSystemRunner` | Wires this model into a `HelixDualSystemRunner` for streaming 50 Hz S1 control with periodic S2 re-invocation. |
| `EmbedInstructionTokens(Tensor<>)` | Looks up instruction-token embeddings through the learned `_tokenEmbedding` table (NVIDIA GR00T N1 2025, §3.1). |
| `GetParameters` |  |
| `PredictAction(Tensor<>,String)` | Predicts a continuous joint-command horizon using the GR00T N1 dual-system protocol: one S2 pass (Eagle-2 VLM) + flow-matching Euler integration via `ActionHead`. |
| `SetParameters(Vector<>)` |  |
| `System1Velocity(Tensor<>,Double,Tensor<>)` | System-1 velocity field: takes a noisy action tensor at flow time `t` and the System-2 latent, runs the DiT-style velocity network, and returns the per-dim velocity. |
| `System2Forward(Tensor<>,String)` | System-2 forward pass: SigLIP-style vision encoder + Eagle-2 LLM decoder + latent projection (paper §3.1). |

