---
title: "MRMR<T>"
description: "Minimum Redundancy Maximum Relevance (mRMR) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Minimum Redundancy Maximum Relevance (mRMR) for feature selection.

## For Beginners

mRMR tries to find features that are both useful and
non-redundant. If two features tell you basically the same thing, you only need
one of them. This method picks features that give you new, useful information
rather than repeating what you already know.

## How It Works

mRMR selects features that have high mutual information with the target (relevance)
while having low mutual information with already selected features (redundancy).

