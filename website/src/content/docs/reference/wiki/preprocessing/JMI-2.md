---
title: "JMI<T>"
description: "Joint Mutual Information (JMI) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Joint Mutual Information (JMI) for feature selection.

## For Beginners

JMI doesn't just look at each feature individually; it
considers how pairs of features work together. When selecting a new feature, it asks:
"How much new information does this feature provide about the target when combined
with each already-selected feature?" This helps find features that complement each other.

## How It Works

JMI selects features by maximizing the joint mutual information between candidate
features and already-selected features with respect to the target. It considers
second-order feature interactions for improved selection.

