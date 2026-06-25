---
title: "MinimalRedundancy<T>"
description: "Minimal Redundancy feature selection for unsupervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Minimal Redundancy feature selection for unsupervised learning.

## For Beginners

If two features are almost identical (highly
correlated), keeping both is wasteful. This method picks features that are
as different from each other as possible, giving you maximum diversity.

## How It Works

Minimal Redundancy selects a diverse set of features that are not highly
correlated with each other. This ensures the selected features provide
complementary rather than overlapping information.

