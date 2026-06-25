---
title: "SignificanceAnalysis<T>"
description: "Significance Analysis of Microarrays (SAM) for bioinformatics feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bioinformatics`

Significance Analysis of Microarrays (SAM) for bioinformatics feature selection.

## For Beginners

When analyzing gene expression data, some genes have
very small changes but even smaller variance, making them appear significant by
chance. SAM adds a small buffer to prevent these false alarms. It also uses
shuffling to estimate how many of our "discoveries" might be false.

## How It Works

SAM uses a modified t-statistic that includes a small constant (fudge factor) in
the denominator to avoid false positives from genes with very low variance. It
estimates false discovery rate (FDR) using permutation testing.

