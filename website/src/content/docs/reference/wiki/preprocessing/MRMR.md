---
title: "MRMR<T>"
description: "Minimum Redundancy Maximum Relevance (mRMR) feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bioinformatics`

Minimum Redundancy Maximum Relevance (mRMR) feature selection.

## For Beginners

Imagine choosing team members: you want people with
the right skills (relevance to the task) but with different specialties so they
don't overlap (minimum redundancy). mRMR finds features that are predictive of
the target but provide diverse, non-overlapping information.

## How It Works

mRMR selects features that have high relevance to the target (maximum relevance)
while minimizing redundancy among selected features. Originally developed for
gene expression analysis.

