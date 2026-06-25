---
title: "SelectFPR<T>"
description: "Select features based on False Positive Rate threshold."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Select features based on False Positive Rate threshold.

## For Beginners

This method uses statistical testing to find features
that are unlikely to be uninformative by chance. If you set alpha=0.05, it keeps
features where there's less than 5% chance the feature-target relationship is random.
However, when testing many features, some false positives slip through.

## How It Works

Selects features with p-values below a specified alpha threshold. Does not control
for multiple testing, so at alpha=0.05, approximately 5% of null (non-informative)
features will be incorrectly selected.

