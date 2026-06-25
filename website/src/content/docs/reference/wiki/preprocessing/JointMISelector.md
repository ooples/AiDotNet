---
title: "JointMISelector<T>"
description: "Joint Mutual Information Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Multivariate`

Joint Mutual Information Feature Selection.

## For Beginners

Regular mutual information looks at features one
at a time. Joint MI considers how features work together. It picks features
that provide unique information about the target, avoiding features that just
repeat what others already tell you.

## How It Works

Selects features by maximizing joint mutual information with the target while
minimizing redundancy among selected features.

