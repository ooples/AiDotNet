---
title: "JointMutualInformation<T>"
description: "Joint Mutual Information (JMI) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Joint Mutual Information (JMI) for feature selection.

## For Beginners

Instead of just picking features that are individually
informative, JMI picks features that work well together. Two features might each
be only moderately useful alone, but together they might be very powerful.

## How It Works

JMI maximizes the joint mutual information between selected features and the target.
It considers both the relevance of individual features and their complementary
information when combined with already selected features.

