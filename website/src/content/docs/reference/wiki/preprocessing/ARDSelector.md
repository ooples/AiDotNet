---
title: "ARDSelector<T>"
description: "Automatic Relevance Determination (ARD) Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bayesian`

Automatic Relevance Determination (ARD) Feature Selection.

## For Beginners

ARD is like giving each feature its own "importance dial."
During training, the method automatically turns down the dial for irrelevant features
(making their effect nearly zero) while keeping the dial up for important ones.
Features whose dials get turned all the way down are removed.

## How It Works

Uses Bayesian ARD to automatically determine which features are relevant
by learning individual precision (inverse variance) parameters for each
feature's coefficient. Features with high precision (low variance) are
effectively pruned.

