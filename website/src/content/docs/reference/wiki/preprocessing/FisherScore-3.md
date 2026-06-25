---
title: "FisherScore<T>"
description: "Fisher Score for class-separability based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral`

Fisher Score for class-separability based feature selection.

## For Beginners

A good feature for classification should have
values that are similar within each class but different across classes.
Fisher Score captures this by comparing spread between vs within groups.

## How It Works

Fisher Score measures how well a feature separates different classes.
It's the ratio of between-class variance to within-class variance.
Higher scores indicate better class discrimination.

