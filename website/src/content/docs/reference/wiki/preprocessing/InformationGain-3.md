---
title: "InformationGain<T>"
description: "Information Gain and Gain Ratio feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Information Gain and Gain Ratio feature selection.

## For Beginners

Information Gain measures how much a feature "tells you"
about the target variable. If knowing the feature value greatly reduces your
uncertainty about the outcome, it has high information gain.

Gain Ratio corrects for features with many categories (like IDs) that would
otherwise appear artificially important.

## How It Works

Information Gain measures how much knowing a feature reduces uncertainty about the target.
Gain Ratio normalizes by the intrinsic value to handle features with many values.

Information Gain = H(Y) - H(Y|X)
Gain Ratio = Information Gain / Intrinsic Value

