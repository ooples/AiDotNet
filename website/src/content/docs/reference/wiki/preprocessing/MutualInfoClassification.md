---
title: "MutualInfoClassification<T>"
description: "Mutual Information for classification-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Classification`

Mutual Information for classification-based feature selection.

## For Beginners

Mutual information measures how much knowing
a feature's value helps predict the class. Unlike correlation, MI can detect
any type of relationship, not just linear ones. A high MI score means the
feature contains useful information for classification.

## How It Works

Mutual Information Classification estimates the mutual information between
each feature and the discrete class label. MI measures how much knowing
the feature reduces uncertainty about the class.

