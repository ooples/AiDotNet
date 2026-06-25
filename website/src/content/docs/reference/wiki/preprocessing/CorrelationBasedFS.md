---
title: "CorrelationBasedFS<T>"
description: "Correlation-based Feature Selection (CFS)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Correlation-based Feature Selection (CFS).

## For Beginners

CFS looks for features that are strongly related
to what you want to predict but not too similar to each other. Having two
features that are almost identical doesn't help much - you want diverse
information sources that all point toward the target.

## How It Works

CFS evaluates feature subsets by considering both feature-target correlation
(relevance) and feature-feature correlation (redundancy). Good subsets have
high correlation with the target but low intercorrelation among features.

