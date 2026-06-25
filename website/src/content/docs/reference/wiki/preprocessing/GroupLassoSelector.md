---
title: "GroupLassoSelector<T>"
description: "Group Lasso feature selection for selecting groups of related features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Group Lasso feature selection for selecting groups of related features.

## For Beginners

Sometimes features come in natural groups. For example,
if you have a categorical variable converted to multiple dummy columns, you want
to keep or remove ALL columns for that category together. Group Lasso ensures
related features are selected or dropped as a unit.

## How It Works

Extends Lasso to select entire groups of features together rather than individual
features. Useful when features naturally belong to groups (e.g., dummy variables
for a categorical feature, or related gene pathways).

