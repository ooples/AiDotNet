---
title: "AdaptiveFeatureSelector<T>"
description: "Adaptive Feature Selector that adjusts selection based on performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Incremental`

Adaptive Feature Selector that adjusts selection based on performance.

## For Beginners

Think of this like a sports team manager who
keeps track of how well each player performs. Players who consistently
do well stay on the team, while underperformers get replaced. Over time,
this builds an optimal team (set of features).

## How It Works

This selector adapts its feature selection over time based on observed
performance. Features that consistently perform well are retained, while
poor performers are replaced with alternatives.

