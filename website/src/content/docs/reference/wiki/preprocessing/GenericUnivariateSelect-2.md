---
title: "GenericUnivariateSelect<T>"
description: "Generic univariate feature selector with configurable selection mode."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection`

Generic univariate feature selector with configurable selection mode.

## For Beginners

This is like a Swiss Army knife for feature selection.
Instead of using different classes for different strategies, you pick a mode:

- KBest: "Give me the top 10 features"
- Percentile: "Give me the top 10% of features"
- FPR/FDR/FWE: "Give me features with statistical significance below 0.05"

## How It Works

GenericUnivariateSelect is a flexible feature selector that can operate in different
modes: k-best, percentile, FPR (false positive rate), FDR (false discovery rate),
or FWE (family-wise error rate). This provides a unified interface for various
univariate selection strategies.

