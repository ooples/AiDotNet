---
title: "MultinomialDiffusion<T>"
description: "Implements the multinomial diffusion process for categorical features in TabDDPM."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Implements the multinomial diffusion process for categorical features in TabDDPM.

## For Beginners

This handles the "category" part of the diffusion process.

Forward (corrupting categories):

- Start with a real category (e.g., color = "red")
- At each step, there's a small chance the category "flips" to a random one
- After many steps, the category is essentially random (equally likely to be any color)

Reverse (recovering categories):

- Start with a random category
- The model predicts the probability of each possible category
- Sample from those probabilities to get a cleaner category
- After many steps, you get a realistic category assignment

The noise schedule controls how quickly categories become randomized.

## How It Works

Multinomial diffusion operates on categorical (one-hot) features:

- **Forward process**: Gradually mixes the true category distribution toward uniform

q(x_t | x_0) = (1 - beta_t) * x_{t-1} + beta_t * (1/K) where K is the number of categories

- **Training**: The model predicts log-probabilities of the original categories
- **Loss**: KL divergence between predicted posterior and true posterior
- **Reverse process**: Sample categories from the predicted probability distribution

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultinomialDiffusion(Int32,Double,Double,Random)` | Initializes a new multinomial diffusion process. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumTimesteps` | Gets the number of diffusion timesteps. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddNoise(Vector<>,Int32)` | Adds categorical noise to a one-hot vector at timestep t (forward process). |
| `ComputeLoss(Vector<>,Vector<>,Vector<>,Int32)` | Computes the KL divergence loss between predicted and true posterior. |
| `ComputeLossGradient(Vector<>,Vector<>)` | Computes the gradient of the loss with respect to predicted log-probabilities. |
| `DenoisingStep(Vector<>,Vector<>,Int32)` | Performs one reverse diffusion step for categorical features. |
| `SampleTimestep` | Samples a random timestep uniformly from [0, numTimesteps). |

