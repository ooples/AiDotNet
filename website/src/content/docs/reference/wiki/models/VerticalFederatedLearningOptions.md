---
title: "VerticalFederatedLearningOptions"
description: "Top-level configuration for vertical federated learning (VFL)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Top-level configuration for vertical federated learning (VFL).

## For Beginners

Vertical Federated Learning allows multiple parties that hold
different features for the same entities to jointly train a model without sharing their
raw data. For example, a bank (income, credit score) and a hospital (diagnoses, prescriptions)
can jointly predict loan default risk.

## How It Works

This is fundamentally different from horizontal FL (where each party has the same
features for different samples). VFL requires:

Example:

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch training. |
| `EnableLabelDifferentialPrivacy` | Gets or sets whether to apply differential privacy noise to the label holder's gradients. |
| `EncryptGradients` | Gets or sets whether to encrypt gradients exchanged between parties. |
| `EntityAlignment` | Gets or sets options for Private Set Intersection used to align entities across parties. |
| `LabelDpDelta` | Gets or sets the differential privacy delta for label protection. |
| `LabelDpEpsilon` | Gets or sets the differential privacy epsilon for label protection. |
| `LearningRate` | Gets or sets the learning rate for model training. |
| `MissingFeatures` | Gets or sets options for handling missing features across parties. |
| `NumberOfEpochs` | Gets or sets the number of training epochs. |
| `NumberOfParties` | Gets or sets the number of parties participating in VFL. |
| `RandomSeed` | Gets or sets the random seed for reproducible training. |
| `SplitModel` | Gets or sets options for the split neural network architecture. |
| `Unlearning` | Gets or sets options for GDPR-compliant entity unlearning. |
| `VerboseLogging` | Gets or sets whether to log detailed training metrics (loss, alignment stats, etc.) at each epoch. |

