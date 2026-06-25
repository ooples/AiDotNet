---
title: "MultiLabelMutualInformation<T>"
description: "Multi-Label Mutual Information for multi-label classification feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.MultiLabel`

Multi-Label Mutual Information for multi-label classification feature selection.

## For Beginners

In multi-label classification, each sample can
have multiple labels at once. This method finds features that are informative
for predicting multiple labels, not just one.

## How It Works

Extends mutual information to handle multiple target labels simultaneously.
Aggregates MI scores across all labels to identify features relevant to
multiple prediction targets.

