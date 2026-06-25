---
title: "GrangerCausalitySelector<T>"
description: "Granger Causality-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.TimeSeries`

Granger Causality-based Feature Selection.

## For Beginners

Granger causality asks: does knowing a feature's
past values help predict the target better than just knowing the target's own
past? Features that "Granger-cause" the target have genuine predictive value
for forecasting.

## How It Works

Uses Granger causality tests to select features that help predict the target
beyond what the target's own history can predict.

