---
title: "SelectFDR<T>"
description: "Select features using Benjamini-Hochberg False Discovery Rate control."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Select features using Benjamini-Hochberg False Discovery Rate control.

## For Beginners

When testing many features, some will look significant
by chance. FDR control says "among all the features we select, we want at most
5% (or your alpha) to be false positives." This is less strict than FWER but
more practical for high-dimensional data.

## How It Works

Uses the Benjamini-Hochberg procedure to control the expected proportion of
false discoveries among selected features. This provides more power than
FWER control while still limiting false positives.

