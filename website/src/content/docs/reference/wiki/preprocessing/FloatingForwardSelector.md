---
title: "FloatingForwardSelector<T>"
description: "Sequential Floating Forward Selection (SFFS)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sequential`

Sequential Floating Forward Selection (SFFS).

## For Beginners

Regular forward selection only adds features but never
removes them. SFFS adds features but can also remove them if doing so improves
the result. This "floating" behavior helps find better feature combinations.

## How It Works

Combines forward selection with conditional backward steps to escape local optima,
allowing previously added features to be removed if they become redundant.

