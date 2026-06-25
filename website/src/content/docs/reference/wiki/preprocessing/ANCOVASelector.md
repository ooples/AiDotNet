---
title: "ANCOVASelector<T>"
description: "ANCOVA (Analysis of Covariance) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Statistical`

ANCOVA (Analysis of Covariance) based Feature Selection.

## For Beginners

ANCOVA tests group differences while accounting
for the effect of one or more continuous covariates. This helps isolate the
true effect of categorical grouping by removing the influence of confounding
variables. Features with significant adjusted F-statistics are selected.

## How It Works

Selects features based on ANCOVA, which combines ANOVA with regression
to control for confounding continuous covariates.

