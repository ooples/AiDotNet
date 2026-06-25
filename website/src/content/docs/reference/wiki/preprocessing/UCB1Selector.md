---
title: "UCB1Selector<T>"
description: "UCB1 (Upper Confidence Bound) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bandit`

UCB1 (Upper Confidence Bound) based Feature Selection.

## For Beginners

UCB1 treats feature selection like a slot machine problem.
Each feature is an "arm" to pull. We want features that are good (high reward)
but also want to try less-tested features that might be better. UCB1 picks features
with high estimated value plus an uncertainty bonus. As we "pull" features more,
the uncertainty decreases, and we converge to truly good features.

## How It Works

Selects features using the UCB1 algorithm from multi-armed bandit theory,
balancing exploitation of known good features with exploration of uncertain ones.

