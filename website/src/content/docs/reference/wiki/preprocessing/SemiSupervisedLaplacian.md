---
title: "SemiSupervisedLaplacian<T>"
description: "Semi-Supervised Feature Selection using Laplacian regularization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.SemiSupervised`

Semi-Supervised Feature Selection using Laplacian regularization.

## For Beginners

When you have few labeled samples but many
unlabeled ones, this method uses both. It builds a neighborhood graph
from all data to understand the data's structure, while using labels
where available.

## How It Works

Combines supervised information from labeled samples with the manifold
structure captured from all samples (labeled and unlabeled) using the
graph Laplacian.

