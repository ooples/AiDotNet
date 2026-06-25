---
title: "SAM<T>"
description: "Significance Analysis of Microarrays (SAM) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bioinformatics`

Significance Analysis of Microarrays (SAM) for feature selection.

## For Beginners

SAM was designed for gene expression data where we
want to find genes that behave differently between conditions (e.g., healthy vs
disease). It uses permutation testing to estimate how many "significant" genes
we'd find just by chance, helping control false positives.

## How It Works

SAM uses a modified t-statistic with a fudge factor to identify significantly
differentially expressed genes. It controls the false discovery rate (FDR)
using permutation-based estimation.

