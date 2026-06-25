---
title: "HotellingT2Selector<T>"
description: "Hotelling's T² based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

Hotelling's T² based Feature Selection.

## For Beginners

While a t-test compares means of one variable,
Hotelling's T² compares mean vectors of multiple variables simultaneously.
It accounts for correlations between features when testing if two groups differ.
Features contributing most to group separation are selected.

## How It Works

Selects features based on Hotelling's T² statistic, the multivariate
generalization of the t-test for comparing two group means.

