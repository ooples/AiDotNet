---
title: "INoiseScheduler<T>"
description: "Interface for diffusion model noise schedulers that control the noise schedule during inference."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for diffusion model noise schedulers that control the noise schedule during inference.

## For Beginners

Think of a noise scheduler like a recipe for gradually revealing a hidden picture.

Imagine you have a clear photograph that you've covered with many layers of static (noise).
The scheduler tells you:

- How many layers of static there are (timesteps)
- How much static is in each layer (noise schedule)
- How to remove one layer at a time to gradually reveal the picture (step function)

Different schedulers (DDIM, PNDM, DPM-Solver) are like different techniques for removing
the static - some are faster, some produce better quality, and some offer a tradeoff.

Key concepts:

- Timesteps: Discrete steps in the noise schedule (e.g., 1000 training steps, 50 inference steps)
- Beta schedule: Controls how much noise is added at each step
- Step function: Takes a noisy sample and model prediction, returns a slightly less noisy sample

## How It Works

Noise schedulers are a core component of diffusion models that control how noise is gradually
added to or removed from data during the diffusion process. They define the noise schedule
(how much noise at each timestep) and provide the mathematical operations to denoise samples.

**Note:** This interface was renamed from IStepScheduler to INoiseScheduler to avoid
confusion with learning rate schedulers (ILearningRateScheduler). Noise schedulers are
specific to diffusion models, while learning rate schedulers control optimization dynamics.

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Gets the scheduler configuration. |
| `Timesteps` | Gets the timesteps for the current inference schedule. |
| `TrainTimesteps` | Gets the number of training timesteps this scheduler was configured with. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Vector<>,Vector<>,Int32)` | Adds noise to a clean sample according to the noise schedule. |
| `GetAlphaCumulativeProduct(Int32)` | Gets the cumulative product of alphas (signal retention) at a given timestep. |
| `GetState` | Gets the current scheduler state for checkpointing. |
| `LoadState(Dictionary<String,Object>)` | Loads scheduler state from a checkpoint. |
| `SetTimesteps(Int32)` | Sets up the inference timesteps based on the number of steps desired. |
| `Step(Vector<>,Int32,Vector<>,,Vector<>)` | Performs one denoising step using the model output. |

