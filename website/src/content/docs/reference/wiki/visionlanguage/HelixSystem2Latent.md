---
title: "HelixSystem2Latent<T>"
description: "Conditioning signal that Helix's slow System-2 VLM produces for the fast System-1 controller."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

Conditioning signal that Helix's slow System-2 VLM produces for the fast System-1 controller.

## How It Works

Per Helix (Figure AI, 2025, "Helix: A Vision-Language-Action Model for Generalist Humanoid Control",
arXiv:2502.07092) the dual-system architecture splits responsibilities:

This class captures the per-tick S2 latent (timestamp, latent tensor, freshness counter) so the
dual-system runner can decide whether to re-invoke S2 or simply reuse the cached latent for the
next S1 step.

## Properties

| Property | Summary |
|:-----|:--------|
| `Latent` | The latent feature vector produced by S2 — typically `DecoderDim`-sized. |
| `ProducedAtTick` | S1 tick on which this latent was produced. |
| `ValidForTicks` | Maximum number of S1 ticks this latent remains valid; after that the runner must produce a fresh one. |

## Methods

| Method | Summary |
|:-----|:--------|
| `IsStaleAt(Int32)` | Returns true when the latent has expired and S2 must be re-invoked. |
| `L2Norm` | Returns the L2-norm of the latent — useful for monitoring "S2 confidence" between invocations. |

