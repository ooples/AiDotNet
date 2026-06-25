---
title: "EnsembleFS<T>"
description: "Ensemble Feature Selection combining multiple methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Ensemble Feature Selection combining multiple methods.

## For Beginners

Different feature selection methods have different
biases and catch different patterns. By combining multiple methods (like asking
multiple experts), we get a more reliable answer. Features that everyone agrees
are important are more trustworthy than those only one method likes.

## How It Works

Ensemble Feature Selection combines the results of multiple feature selection
methods using voting or rank aggregation. Features consistently selected across
multiple methods are more likely to be truly important.

