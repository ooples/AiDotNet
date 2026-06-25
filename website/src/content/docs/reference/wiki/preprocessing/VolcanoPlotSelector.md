---
title: "VolcanoPlotSelector<T>"
description: "Volcano plot-based feature selection combining fold change and statistical significance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bioinformatics`

Volcano plot-based feature selection combining fold change and statistical significance.

## For Beginners

A feature might change a lot but by random chance,
or change little but reliably. Volcano selection requires BOTH: the feature must
show a large difference between groups AND that difference must be statistically
significant (unlikely to be just noise).

## How It Works

Selects features that show both large fold changes AND statistical significance.
Named after the volcano-shaped plot with log fold change on x-axis and
-log(p-value) on y-axis.

