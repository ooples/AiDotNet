---
title: "SparseAutoencoderSelector<T>"
description: "Sparse Autoencoder-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Neural`

Sparse Autoencoder-based Feature Selection.

## For Beginners

A sparse autoencoder forces most of its internal
neurons to be "off" most of the time. This makes it focus on the most
important input features. Features that activate the sparse neurons are
the ones we select.

## How It Works

Uses a sparse autoencoder with sparsity constraints on the hidden layer
to learn which input features are essential for reconstruction.

