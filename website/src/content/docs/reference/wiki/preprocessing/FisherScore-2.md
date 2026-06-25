---
title: "FisherScore<T>"
description: "Fisher Score for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Fisher Score for feature selection.

## For Beginners

Imagine you're sorting fruits by color. A good feature (like
color) would make all apples similar to each other but different from oranges. Fisher
Score measures exactly this: how much a feature groups similar items together while
keeping different groups apart.

## How It Works

Fisher Score measures the ratio of between-class variance to within-class variance.
Features with high Fisher Scores have good class separability, meaning samples from
different classes are far apart while samples from the same class are close together.

