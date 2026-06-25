---
title: "EmbeddedWrapperHybridSelector<T>"
description: "Embedded-Wrapper Hybrid Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Embedded-Wrapper Hybrid Feature Selection.

## For Beginners

This method first uses Lasso regression to
automatically shrink unimportant features to zero, giving us a candidate
set. Then it fine-tunes this selection by trying to add or remove features
to improve model performance. This balances speed with accuracy.

## How It Works

Combines embedded (L1 regularization) and wrapper methods: first uses
Lasso to identify a candidate set, then refines selection using sequential
feature addition/removal.

