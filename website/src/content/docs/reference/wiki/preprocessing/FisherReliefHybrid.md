---
title: "FisherReliefHybrid<T>"
description: "Fisher-Relief Hybrid feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Fisher-Relief Hybrid feature selection.

## For Beginners

Fisher Score looks at class averages while
ReliefF looks at individual sample relationships. Combining them gives
you features that work well from both viewpoints - good class separation
AND good at distinguishing similar vs different samples.

## How It Works

Combines Fisher Score (parametric, class separability) with ReliefF
(instance-based, neighbor relationships) to leverage both perspectives.

