---
title: "MIFS<T>"
description: "Mutual Information Feature Selection (MIFS) with redundancy penalty."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Mutual Information Feature Selection (MIFS) with redundancy penalty.

## For Beginners

MIFS greedily picks features that tell you a lot about
the target. But it also subtracts a penalty if the new feature overlaps with what
you already know from selected features. The beta parameter controls how harsh
this penalty is: 0 = no penalty (just pick high MI), 1 = full penalty.

## How It Works

MIFS iteratively selects features that have high mutual information with the target
while penalizing redundancy with already-selected features. The beta parameter
controls the strength of the redundancy penalty.

