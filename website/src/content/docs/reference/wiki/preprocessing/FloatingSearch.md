---
title: "FloatingSearch<T>"
description: "Sequential Floating Forward Selection (SFFS) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Sequential Floating Forward Selection (SFFS) for feature selection.

## For Beginners

Regular forward selection adds features one
at a time but never reconsiders its choices. SFFS can "float" - after
adding a feature, it checks if any previously added features are now
unnecessary and removes them. This leads to better final selections.

## How It Works

SFFS is an advanced wrapper method that performs forward selection with
conditional backward elimination. After each forward step, it attempts
to remove features that may have become redundant.

