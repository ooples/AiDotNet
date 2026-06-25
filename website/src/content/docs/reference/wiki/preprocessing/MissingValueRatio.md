---
title: "MissingValueRatio<T>"
description: "Missing Value Ratio filter for removing features with too many missing values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Unsupervised`

Missing Value Ratio filter for removing features with too many missing values.

## For Beginners

Features with many missing values are unreliable.
If more than X% of a feature's values are missing, it's often better to
remove it entirely rather than try to impute the missing data.

## How It Works

Removes features where the proportion of missing values exceeds a threshold.
Missing values are identified as NaN, infinity, or optionally a specified
sentinel value.

