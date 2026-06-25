---
title: "PATEGANOptions<T>"
description: "Configuration options for PATE-GAN, a differentially private GAN that uses the Private Aggregation of Teacher Ensembles (PATE) framework for privacy-preserving synthetic data generation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for PATE-GAN, a differentially private GAN that uses the
Private Aggregation of Teacher Ensembles (PATE) framework for privacy-preserving
synthetic data generation.

## For Beginners

PATE-GAN provides privacy by never showing the full dataset to any
single component:

1. Split real data into N groups, give each to a "teacher"
2. Teachers vote on whether generated data looks real or fake
3. Add random noise to the vote tally (for privacy)
4. A "student" learns from the noisy votes, not the real data
5. Generator tries to fool the student

Because the student never sees real data, and the votes are noisy,
no individual's data can be extracted from the generated output.

Example:

## How It Works

PATE-GAN achieves differential privacy through a teacher-student framework:

- **Teacher ensemble**: Multiple discriminators, each trained on a disjoint partition of the data
- **Noisy aggregation**: Teacher votes are aggregated with Laplace noise
- **Student discriminator**: Learns from noisy teacher labels, never sees real data directly
- **Generator**: Trained against the student discriminator

Reference: "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees"
(Jordon et al., ICLR 2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `EmbeddingDimension` | Gets or sets the dimension of the random noise vector for the generator. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GeneratorDimensions` | Gets or sets the hidden layer sizes for the generator network. |
| `LaplaceScale` | Gets or sets the Laplace noise scale for the noisy aggregation mechanism. |
| `LearningRate` | Gets or sets the learning rate. |
| `NumTeachers` | Gets or sets the number of teacher discriminators in the ensemble. |
| `StudentDimensions` | Gets or sets the hidden layer sizes for the student discriminator. |
| `StudentDropout` | Gets or sets the dropout rate for the student discriminator hidden layers. |
| `StudentSteps` | Gets or sets the number of student discriminator training steps per teacher query. |
| `TeacherDimensions` | Gets or sets the hidden layer sizes for each teacher discriminator. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column transformation. |

