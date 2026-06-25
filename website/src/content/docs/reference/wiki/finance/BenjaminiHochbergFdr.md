---
title: "BenjaminiHochbergFdr"
description: "Benjamini-Hochberg false-discovery-rate (FDR) control for multiple hypothesis testing."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Evaluation`

Benjamini-Hochberg false-discovery-rate (FDR) control for multiple hypothesis testing. Given a set of
p-values and a target FDR level alpha, it decides which hypotheses to reject and returns BH-adjusted
q-values.

## For Beginners

Say you backtest 100 signals and 5 look "significant" at the usual p < 0.05.
But testing 100 things means several will look good by pure chance. BH-FDR is a smarter, less strict
cousin of the Bonferroni correction: instead of demanding every signal clear an impossibly high bar, it
keeps the *fraction* of bogus discoveries among your accepted signals below a level you choose
(say 10%). The q-value is "the smallest FDR at which this hypothesis would still be accepted" — handy
for ranking discoveries.

## How It Works

When many strategies / signals / features are each tested for significance, the chance of at least one
false positive explodes. The Benjamini-Hochberg procedure controls the expected proportion of false
discoveries among the rejected hypotheses at level alpha: sort the p-values ascending, find the largest
rank k with p_(k) ≤ (k / m)·alpha, and reject all hypotheses with rank ≤ k. The q-value is the
monotone-adjusted p-value (min over later ranks of m/rank · p), capped at 1.

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply(IReadOnlyList<Double>,Double)` | Runs the Benjamini-Hochberg FDR procedure. |

