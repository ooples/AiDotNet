---
title: "AutoencoderSelector<T>"
description: "Autoencoder based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Neural`

Autoencoder based Feature Selection.

## For Beginners

An autoencoder compresses data and reconstructs it.
Features that are harder to reconstruct accurately are often more important
because they carry unique information not captured by other features.

## How It Works

Selects features based on their reconstruction importance in a simple autoencoder,
measuring how much each feature contributes to the learned representation.

