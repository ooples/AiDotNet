---
title: "HelixDualSystemRunner<T>"
description: "Streaming controller that coordinates Helix's fast System-1 visuomotor policy (200 Hz) and slow System-2 VLM (7–9 Hz) per Figure AI's dual-rate dual-system architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

Streaming controller that coordinates Helix's fast System-1 visuomotor policy (200 Hz) and
slow System-2 VLM (7–9 Hz) per Figure AI's dual-rate dual-system architecture.

## How It Works

Per Helix (Figure AI 2025, arXiv:2502.07092), high-level intent runs slowly while low-level
control runs fast. This runner is the explicit coordinator: each call to `String)`
advances one S1 tick and lazily re-invokes S2 when the cached latent has gone stale (every
`System2TicksValid` S1 ticks — default 22 to match S1:S2 = 200Hz : ~9Hz).

The runner is generic over the model's S2 and S1 callbacks so the same coordination logic is
reusable: `Helix` wires its native VLM into `system2Forward` and its action
head into `system1Forward`. Future Helix-class models (e.g. Figure's later iterations) can
reuse this runner unchanged.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HelixDualSystemRunner(Func<Tensor<>,String,Tensor<>>,Func<Tensor<>,Tensor<>,Tensor<>>,Int32)` | Constructs a runner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CachedLatent` | The currently cached S2 latent, or null if no S1 step has been taken yet. |
| `CurrentTick` | Current S1 tick counter since the runner was created or `Reset` was called. |
| `System2TicksValid` | Number of S1 ticks an S2 latent remains valid (default 22 — paper §4.1: S1 @ 200 Hz, S2 @ 7–9 Hz). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Reset` | Resets the tick counter and invalidates the cached S2 latent. |
| `Rollout(Tensor<>,String,Int32)` | Runs `numSteps` consecutive S1 ticks against the same observation/instruction (e.g. |
| `Step(Tensor<>,String)` | Advances one S1 tick. |

