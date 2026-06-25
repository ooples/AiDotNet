---
title: "PointBiserialSelector<T>"
description: "Point-Biserial Correlation Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

Point-Biserial Correlation Feature Selection.

## For Beginners

Point-biserial correlation measures how well
a continuous feature separates two groups (like positive vs negative cases).
Features with high point-biserial correlation have very different values
for the two classes, making them useful for classification.

## How It Works

Uses point-biserial correlation for feature selection when the target
is binary. This is the correlation between a continuous variable and
a binary variable.

