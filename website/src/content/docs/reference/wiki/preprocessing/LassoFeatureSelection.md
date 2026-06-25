---
title: "LassoFeatureSelection<T>"
description: "L1 (Lasso) regularization-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

L1 (Lasso) regularization-based feature selection.

## For Beginners

Lasso is like a strict editor that removes unnecessary words
from a sentence. The L1 penalty forces some feature weights to become exactly zero,
automatically removing unimportant features. Stronger regularization means more features
get eliminated, leaving only the most essential ones.

## How It Works

Lasso regression uses L1 regularization which drives some coefficients exactly to zero,
effectively performing feature selection. Features with non-zero coefficients are selected.
The regularization strength controls how many features are zeroed out.

