---
title: "StreamingFeatureSelector<T>"
description: "Streaming Feature Selector for online/incremental learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Incremental`

Streaming Feature Selector for online/incremental learning.

## For Beginners

When you can't store all your data (it's too big
or comes continuously), you need to update feature scores incrementally.
This method keeps running averages that update with each new sample,
allowing feature selection on endless data streams.

## How It Works

Maintains running statistics to update feature scores as new data arrives.
This enables feature selection on streaming data without storing all
historical samples.

