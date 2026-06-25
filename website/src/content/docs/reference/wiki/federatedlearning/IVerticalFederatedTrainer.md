---
title: "IVerticalFederatedTrainer<T>"
description: "Orchestrates the vertical federated learning training process."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Vertical`

Orchestrates the vertical federated learning training process.

## For Beginners

This is the main coordinator for vertical FL training.
It handles the entire pipeline:

## How It Works

Unlike horizontal FL trainers (which average model parameters), the VFL trainer
coordinates activations flowing through a split neural network. No party ever sees
another party's raw features.

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignEntities(PsiOptions)` | Performs entity alignment across all registered parties using PSI. |
| `Predict(IReadOnlyList<Int32>)` | Makes predictions on aligned entity data using the split model. |
| `RegisterParty(IVerticalParty<>)` | Registers a party to participate in VFL training. |
| `Train` | Runs the full training loop for the configured number of epochs. |
| `TrainEpoch` | Runs a single training epoch over the aligned data. |
| `UnlearnEntities(IReadOnlyList<String>)` | Removes the influence of specified entities from the trained model (unlearning). |

