---
title: "NMFBasedSelector<T>"
description: "Non-negative Matrix Factorization (NMF) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Projection`

Non-negative Matrix Factorization (NMF) based Feature Selection.

## For Beginners

NMF breaks down your data into parts (like
breaking a photo into its component patterns). Features that contribute
most to these fundamental patterns are the most informative ones to keep.
NMF is especially good when features represent non-negative quantities.

## How It Works

Uses NMF to decompose the data matrix and selects features based on
their contribution to the learned basis vectors.

