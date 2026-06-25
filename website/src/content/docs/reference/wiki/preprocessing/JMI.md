---
title: "JMI<T>"
description: "Joint Mutual Information (JMI) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Joint Mutual Information (JMI) for feature selection.

## For Beginners

JMI looks at how much a feature tells you about
the target when combined with previously selected features. A feature might
be weak alone but very powerful in combination with others. JMI finds these
synergistic feature combinations.

## How It Works

JMI selects features that maximize joint mutual information with the target,
considering both the individual feature-target relationship and how features
work together.

