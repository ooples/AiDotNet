---
title: "ISemiSupervisedClassifier<T>"
description: "Defines the interface for semi-supervised classification algorithms that can learn from both labeled and unlabeled data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the interface for semi-supervised classification algorithms that can learn from both labeled and unlabeled data.

## For Beginners

Imagine you're teaching someone to recognize different types of flowers.

In traditional supervised learning, you'd need to label every single flower image: "This is a rose",
"This is a tulip", etc. But labeling thousands of images is time-consuming and expensive.

Semi-supervised learning is smarter - it uses a small set of labeled examples (maybe 100 labeled flowers)
combined with a large set of unlabeled images (maybe 10,000 unlabeled flower photos). The algorithm
learns patterns from the unlabeled data to improve its predictions.

Real-world applications include:

- Medical diagnosis: Only a few X-rays have expert diagnoses, but many are unlabeled
- Document classification: A few documents are categorized, thousands are not
- Speech recognition: Limited transcribed audio, abundant raw recordings

This interface extends IClassifier, meaning all semi-supervised classifiers can also be used
as regular classifiers and inherit all the IFullModel capabilities like serialization and checkpointing.

## How It Works

Semi-supervised learning is a machine learning paradigm that combines a small amount of labeled data
with a large amount of unlabeled data during training. This approach can significantly improve
learning accuracy when labeled data is scarce or expensive to obtain.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumLabeledSamples` | Gets the number of labeled samples used in training. |
| `NumUnlabeledSamples` | Gets the number of unlabeled samples used in training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetPseudoLabelConfidences` | Gets the confidence scores for the pseudo-labels. |
| `GetPseudoLabels` | Gets the pseudo-labels assigned to the unlabeled data during training. |
| `TrainSemiSupervised(Matrix<>,Vector<>,Matrix<>)` | Trains the classifier using both labeled and unlabeled data. |

