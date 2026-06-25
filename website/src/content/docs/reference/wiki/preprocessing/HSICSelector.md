---
title: "HSICSelector<T>"
description: "Hilbert-Schmidt Independence Criterion (HSIC) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Kernel`

Hilbert-Schmidt Independence Criterion (HSIC) based Feature Selection.

## For Beginners

HSIC is a powerful way to measure if two variables
are related, even in complex non-linear ways. By computing HSIC between each
feature and the target, we find features that are truly informative, even when
the relationship isn't a straight line.

## How It Works

Uses HSIC to measure the statistical dependency between features and target
in a reproducing kernel Hilbert space, capturing non-linear dependencies.

