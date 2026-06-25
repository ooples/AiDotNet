---
title: "WassersteinLoss"
description: "Implements the Wasserstein loss function used in Wasserstein Generative Adversarial Networks (WGAN)."
section: "Reference"
---

_Loss Functions_

Implements the Wasserstein loss function used in Wasserstein Generative Adversarial Networks (WGAN).

## For Beginners

Wasserstein loss is a special way to measure how different two groups of data are.

Why use Wasserstein loss instead of regular binary cross-entropy?

- More stable training - gradients don't vanish when the critic is confident
- The loss value correlates with image quality - lower loss means better images
- No mode collapse - the generator doesn't get stuck producing the same output
- Can train the critic to convergence without breaking training

How it works:

- For real images, we want the critic to output high scores (label = +1)
- For fake images, we want the critic to output low scores (label = -1)
- The loss is simply the average of (score * label)
- A well-trained critic gives positive scores to real images and negative scores to fakes

Reference: Arjovsky et al., "Wasserstein GAN" (2017)

## How It Works

The Wasserstein loss (also known as Earth Mover's Distance loss) measures the distance between
two probability distributions. In the context of GANs, it provides a meaningful gradient signal
even when the discriminator (critic) is well-trained.

**Mathematical Formula:**

- Loss = mean(predicted * label)
- Where label is +1 for real samples, -1 for fake samples
- The critic aims to maximize E[critic(real)] - E[critic(fake)]

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new WassersteinLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"WassersteinLoss = {value:F4}");
```

