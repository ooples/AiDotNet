---
title: "MaximalInformationSelector<T>"
description: "Maximal Information Coefficient (MIC) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Nonparametric`

Maximal Information Coefficient (MIC) based Feature Selection.

## For Beginners

MIC tries different ways of binning data to find
the binning that maximizes mutual information. This makes it very good at
detecting all kinds of relationships - linear, nonlinear, periodic, etc.

## How It Works

Selects features based on their maximal information coefficient with the target,
capturing many different types of relationships including nonlinear ones.

