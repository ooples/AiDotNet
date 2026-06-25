---
title: "JointEntropySelector<T>"
description: "Joint Entropy based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Entropy`

Joint Entropy based Feature Selection.

## For Beginners

Joint entropy measures the total randomness when
considering a feature and target together. Lower joint entropy relative to
individual entropies indicates the feature and target are related.

## How It Works

Selects features based on their joint entropy with the target, favoring
features that share information with the target.

