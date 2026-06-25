---
title: "BorutaSelector<T>"
description: "Boruta feature selection using shadow features and random forest importance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Boruta feature selection using shadow features and random forest importance.

## For Beginners

Imagine testing each feature against a "random noise"
version of itself. If a real feature is important, it should beat its noisy twin
repeatedly. Boruta does this systematically, keeping only features that clearly
outperform random chance.

## How It Works

Boruta creates "shadow" features by shuffling original features. A random forest
is trained, and features that consistently outperform the best shadow feature
are confirmed as important. Features that never beat shadows are rejected.

