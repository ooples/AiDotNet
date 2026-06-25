---
title: "SlidingWindowSelector<T>"
description: "Sliding Window Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Streaming`

Sliding Window Feature Selection.

## For Beginners

Data patterns can change over time (concept drift).
This method only looks at the most recent samples in a sliding window, so it
adapts to changes. Features that were important last month might not be
important now, and this method handles that.

## How It Works

Maintains a sliding window of recent samples and performs feature selection
based on the most recent data, adapting to concept drift.

