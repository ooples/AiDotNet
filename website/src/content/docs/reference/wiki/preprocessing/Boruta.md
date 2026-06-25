---
title: "Boruta<T>"
description: "Boruta feature selection algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bioinformatics`

Boruta feature selection algorithm.

## For Beginners

Boruta asks: "Is this feature better than random noise?"
It shuffles your features to create meaningless versions (shadows), then compares
real features against these shadows. Features that consistently beat their shadows
are truly informative.

## How It Works

Boruta creates "shadow features" (shuffled copies of real features) and compares
feature importance against these shadows. Features consistently outperforming
the best shadow are confirmed as important.

