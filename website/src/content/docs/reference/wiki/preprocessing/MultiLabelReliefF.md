---
title: "MultiLabelReliefF<T>"
description: "Multi-Label ReliefF for multi-label feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.MultiLabel`

Multi-Label ReliefF for multi-label feature selection.

## For Beginners

In multi-label problems (like tagging photos with
multiple keywords), a single image might be "sunset", "beach", and "romantic"
all at once. This algorithm finds features that help distinguish between
different label combinations, not just individual labels.

## How It Works

Multi-Label ReliefF extends the ReliefF algorithm to handle multi-label
classification problems where each instance can belong to multiple classes
simultaneously. It considers label correlations when computing feature weights.

