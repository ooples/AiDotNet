---
title: "VariationalAutoencoderSelector<T>"
description: "Variational Autoencoder-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Neural`

Variational Autoencoder-based Feature Selection.

## For Beginners

A VAE learns not just a compressed representation
but a probability distribution over possible representations. Features that
contribute most to this learned distribution are the important ones.

## How It Works

Uses a Variational Autoencoder (VAE) to learn a probabilistic latent space,
then selects features based on their contribution to the learned distribution.

