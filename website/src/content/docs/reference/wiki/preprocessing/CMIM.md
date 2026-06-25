---
title: "CMIM<T>"
description: "Conditional Mutual Information Maximization (CMIM) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Conditional Mutual Information Maximization (CMIM) for feature selection.

## For Beginners

CMIM picks features that provide new information about
the target that isn't already captured by previously selected features. It's like
building a team where each new member brings truly unique knowledge.

## How It Works

CMIM selects features that maximize mutual information with the target while
conditioning on already selected features. It addresses redundancy by considering
conditional dependencies.

