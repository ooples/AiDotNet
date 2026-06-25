---
title: "BorutaFS<T>"
description: "Boruta algorithm for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Boruta algorithm for feature selection.

## For Beginners

Boruta creates fake versions of your features by randomly
shuffling their values. It then asks: "Is this real feature more useful than the best
fake feature?" Features that consistently outperform their fake counterparts are kept.
This is more rigorous than just picking the top N features.

## How It Works

Boruta creates "shadow" features by shuffling each original feature, then trains
a model to compare original features against their shadows. Features that consistently
beat the best shadow feature are confirmed as important; those that don't are rejected.

