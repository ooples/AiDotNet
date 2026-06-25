---
title: "TraceRatio<T>"
description: "Trace Ratio criterion for multi-class feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral`

Trace Ratio criterion for multi-class feature selection.

## For Beginners

This extends Fisher Score to consider how features
work together. Instead of scoring each feature independently, it evaluates
how a set of features collectively separates classes.

## How It Works

Trace Ratio maximizes the ratio of between-class scatter to within-class
scatter. Unlike Fisher Score which works per feature, Trace Ratio considers
feature interactions through scatter matrices.

