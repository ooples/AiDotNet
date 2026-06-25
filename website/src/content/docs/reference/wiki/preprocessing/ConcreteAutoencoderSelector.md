---
title: "ConcreteAutoencoderSelector<T>"
description: "Concrete Autoencoder Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Neural`

Concrete Autoencoder Feature Selection.

## For Beginners

The Concrete Autoencoder learns which features
to select as part of its training. It uses a special trick (Gumbel-Softmax)
that lets it learn a "soft" selection that gradually becomes "hard" (0 or 1)
as training progresses.

## How It Works

Uses the Concrete Autoencoder which learns a discrete feature selection
mask using the Concrete/Gumbel-Softmax distribution for differentiable
feature selection.

