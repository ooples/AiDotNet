---
title: "ClassWeightedFS<T>"
description: "Class-Weighted Feature Selection for imbalanced datasets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Imbalanced`

Class-Weighted Feature Selection for imbalanced datasets.

## For Beginners

In datasets where one class has few examples,
normal feature selection might favor features that only work for the common class.
This method weights the rare class more heavily, ensuring selected features
can identify both common and rare cases.

## How It Works

Class-Weighted Feature Selection applies class weights when computing feature
scores to give more importance to minority class samples. This helps select
features that are discriminative for underrepresented classes.

