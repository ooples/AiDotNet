---
title: "InformationGain<T>"
description: "Information Gain (Mutual Information) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Information Gain (Mutual Information) for feature selection.

## For Beginners

Information Gain asks: "How much does knowing this
feature reduce my uncertainty about the target?" If knowing the feature value
tells you a lot about what the target will be, it has high information gain.
It's measured in bits (or nats, depending on the logarithm base).

## How It Works

Information Gain measures the reduction in entropy of the target variable when
a feature is known. It's equivalent to mutual information and quantifies how much
information about the target a feature provides.

