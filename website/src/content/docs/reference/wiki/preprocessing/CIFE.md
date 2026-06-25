---
title: "CIFE<T>"
description: "Conditional Infomax Feature Extraction (CIFE) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Conditional Infomax Feature Extraction (CIFE) for feature selection.

## For Beginners

CIFE evaluates how much new information a feature
provides about the target, given what you already know from selected features.
It's like asking "what do you know that I don't already know?" for each feature.

## How It Works

CIFE is an information-theoretic method that maximizes the conditional mutual
information while considering the interaction between features and the target.

