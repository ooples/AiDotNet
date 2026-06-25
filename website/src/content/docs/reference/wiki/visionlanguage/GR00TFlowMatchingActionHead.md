---
title: "GR00TFlowMatchingActionHead<T>"
description: "Flow-matching action head used by GR00T N1 to denoise continuous joint commands conditioned on the System-2 vision-language latent."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.Robotics`

Flow-matching action head used by GR00T N1 to denoise continuous joint commands conditioned on
the System-2 vision-language latent.

## How It Works

Per GR00T N1 (NVIDIA, 2025, "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots",
arXiv:2503.14734), the System-1 action policy is a DiT-style transformer trained with the
**flow-matching** objective (Lipman et al. 2023, arXiv:2210.02747). Inference reverses the
learned vector field by Euler-integrating from Gaussian noise at `t=0` to the data
distribution at `t=1`; each integration step queries the network for the velocity field
`v_θ(x_t, t | latent)` and updates `x_{t+Δt} = x_t + Δt · v_θ`.

This class implements the inference-time integrator. The per-step velocity network is supplied
as a callback (the GR00T-N1 model wires its DiT decoder there), keeping this class composable.

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GR00TFlowMatchingActionHead(Func<Tensor<>,Double,Tensor<>,Tensor<>>,Int32,Nullable<Int32>)` | Builds an action head. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumIntegrationSteps` | Number of Euler integration steps for inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Generate(Int32,Tensor<>)` | Generates an action tensor of length `actionDimension` by Euler-integrating the flow-matching vector field from Gaussian noise at t=0 to the data distribution at t=1, conditioned on `system2Latent`. |
| `GenerateHorizon(Int32,Int32,Tensor<>)` | Generates a horizon of `horizon` action vectors, each of length `actionDimension`, concatenated into a flat tensor. |

