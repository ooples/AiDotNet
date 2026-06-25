---
title: "GaussianDiffusion<T>"
description: "Implements the Gaussian diffusion process for continuous/numerical features in TabDDPM."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Implements the Gaussian diffusion process for continuous/numerical features in TabDDPM.

## For Beginners

This handles the "numbers" part of the diffusion process.

Forward (adding noise to numbers):

- Start with a real number (e.g., salary = $50,000)
- At each step, mix in a little bit of random Gaussian noise
- After 1000 steps, the number is pure noise (could be anything)

Reverse (removing noise from numbers):

- Start with a random number
- The model predicts what noise was added
- Remove that noise to get slightly cleaner number
- After 1000 removals, you have a realistic salary value

The math ensures that each step is a small, reversible change.

## How It Works

The Gaussian diffusion process operates on continuous features:

- **Forward process** (add noise): q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
- **Training**: The model learns to predict the noise epsilon that was added at timestep t
- **Loss**: MSE between predicted and actual noise
- **Reverse process** (denoise): x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * predicted_noise) + sigma_t * z

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianDiffusion(Int32,Double,Double,String,Random)` | Initializes a new Gaussian diffusion process. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumTimesteps` | Gets the number of diffusion timesteps. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Vector<>,Int32,Vector<>)` | Adds noise to clean data at a given timestep (forward process). |
| `ComputeLoss(Vector<>,Vector<>)` | Computes the MSE loss between predicted and actual noise. |
| `ComputeLossGradient(Vector<>,Vector<>)` | Computes the gradient of the MSE loss with respect to predicted noise. |
| `DenoisingStep(Vector<>,Vector<>,Int32)` | Performs one reverse diffusion step (denoising): x_{t-1} from x_t. |
| `SampleTimestep` | Samples a random timestep uniformly from [0, numTimesteps). |

