---
title: "NormalizedMutualInformation<T>"
description: "Normalized Mutual Information for scale-invariant dependency measurement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Normalized Mutual Information for scale-invariant dependency measurement.

## For Beginners

Regular mutual information can be higher for
features with more distinct values. NMI normalizes this so you can fairly
compare features with different numbers of unique values.

## How It Works

Normalized Mutual Information (NMI) divides mutual information by entropy
to get a value between 0 and 1, making comparisons across features more fair.

