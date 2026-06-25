---
title: "MedSynthOptions<T>"
description: "Configuration options for MedSynth, a privacy-preserving medical tabular data synthesis model combining a VAE/GAN hybrid with clinical validity constraints."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for MedSynth, a privacy-preserving medical tabular data synthesis
model combining a VAE/GAN hybrid with clinical validity constraints.

## For Beginners

MedSynth generates fake patient data that is:

1. **Clinically valid**: Lab values are within normal/plausible ranges
2. **Internally consistent**: Related fields make sense together

(e.g., a 5-year-old won't have adult blood pressure values)

3. **Private**: No real patient's data can be extracted

Example:

## How It Works

MedSynth is specialized for medical/health data generation with:

- **Clinical validity constraints**: Ensures generated values fall within valid clinical ranges
- **Differential privacy**: Optional epsilon-differential privacy via gradient clipping and noise
- **Referential integrity**: Maintains consistency between related medical fields
- **Constraint satisfaction layer**: Post-processing to enforce domain rules

## Properties

| Property | Summary |
|:-----|:--------|
| `AdversarialWeight` | Gets or sets the adversarial loss weight. |
| `BatchSize` | Gets or sets the training batch size. |
| `ClipNorm` | Gets or sets the gradient clipping norm for differential privacy. |
| `ConstraintWeight` | Gets or sets the constraint violation penalty weight. |
| `DiscriminatorDimensions` | Gets or sets the hidden layer sizes for the discriminator. |
| `DiscriminatorDropout` | Gets or sets the dropout rate for discriminator hidden layers. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator training steps per VAE/generator step. |
| `EnablePrivacy` | Gets or sets whether to enable differential privacy. |
| `EncoderDimensions` | Gets or sets the hidden layer sizes for the encoder/decoder. |
| `Epochs` | Gets or sets the number of training epochs. |
| `Epsilon` | Gets or sets the privacy budget (epsilon) when privacy is enabled. |
| `KLWeight` | Gets or sets the KL divergence weight. |
| `LatentDimension` | Gets or sets the VAE latent space dimension. |
| `LearningRate` | Gets or sets the learning rate. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column transformation. |

