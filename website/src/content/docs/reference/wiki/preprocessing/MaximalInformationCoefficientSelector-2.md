---
title: "MaximalInformationCoefficientSelector<T>"
description: "Maximal Information Coefficient (MIC) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Nonlinear`

Maximal Information Coefficient (MIC) based Feature Selection.

## For Beginners

MIC is a powerful measure that tries different ways
of binning the data to find the maximum mutual information. It scores relationships
from 0 (independent) to 1 (perfect relationship) regardless of whether the
relationship is linear, exponential, periodic, or complex.

## How It Works

Selects features based on MIC, which finds the maximum mutual information
over different bin configurations, providing equitability across relationship types.

