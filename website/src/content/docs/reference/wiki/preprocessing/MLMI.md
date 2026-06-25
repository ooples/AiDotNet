---
title: "MLMI<T>"
description: "Multi-Label Mutual Information (ML-MI) for multi-label feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.MultiLabel`

Multi-Label Mutual Information (ML-MI) for multi-label feature selection.

## For Beginners

Mutual information measures how much knowing one
thing tells you about another. In multi-label problems, ML-MI measures how
much each feature tells you about the entire combination of labels an instance
might have.

## How It Works

ML-MI extends mutual information to handle multi-label problems by computing
the mutual information between each feature and the entire label set. It can
consider label correlations and dependencies.

