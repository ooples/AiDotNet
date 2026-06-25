---
title: "RFECV<T>"
description: "Recursive Feature Elimination with Cross-Validation (RFECV)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Recursive Feature Elimination with Cross-Validation (RFECV).

## For Beginners

Regular RFE requires you to specify how many
features to keep. RFECV figures out the best number automatically by
testing different counts and seeing which gives the best validation score.

## How It Works

RFECV uses cross-validation to find the optimal number of features.
It performs RFE for different feature counts and selects the count
that maximizes cross-validated performance.

