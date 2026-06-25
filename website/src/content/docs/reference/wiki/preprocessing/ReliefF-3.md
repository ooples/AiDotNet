---
title: "ReliefF<T>"
description: "ReliefF algorithm for feature selection based on nearest neighbor differences."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

ReliefF algorithm for feature selection based on nearest neighbor differences.

## For Beginners

ReliefF picks random samples and looks at their nearest
neighbors. If a feature differs between a sample and its nearby "enemies" (different class)
but stays similar to nearby "friends" (same class), it's useful. The algorithm rewards
features that help separate classes and penalizes those that don't.

## How It Works

ReliefF estimates feature relevance by sampling instances and comparing them to their
nearest neighbors of the same class (hits) and different classes (misses). Features
that differentiate between classes get higher weights.

