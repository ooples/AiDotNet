---
title: "TTestSelector<T>"
description: "Two-Sample T-Test based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

Two-Sample T-Test based Feature Selection.

## For Beginners

The t-test checks if two groups have different
average values. This selector finds features where classes have significantly
different means, which helps in classification tasks.

## How It Works

Selects features based on their t-statistic from two-sample t-tests between
classes, identifying features with significantly different means.

