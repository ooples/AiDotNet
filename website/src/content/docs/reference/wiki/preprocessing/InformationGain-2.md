---
title: "InformationGain<T>"
description: "Information Gain (IG) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Information Gain (IG) for feature selection.

## For Beginners

Entropy measures uncertainty - like not knowing what's
in a wrapped gift. Information Gain tells you how much knowing a feature reduces
that uncertainty. If knowing your age perfectly predicts whether you'll buy
something, then age has high information gain for purchase prediction.

## How It Works

Information Gain measures the reduction in entropy (uncertainty) about the target
variable when a feature is known. Features with high information gain provide the
most information about the target and are useful for prediction.

