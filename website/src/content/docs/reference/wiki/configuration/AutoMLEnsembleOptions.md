---
title: "AutoMLEnsembleOptions"
description: "Configuration options for AutoML ensembling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for AutoML ensembling.

## For Beginners

An ensemble is like asking multiple experts and averaging their answers.
It often performs better than relying on a single model.

## How It Works

Ensembling retrains a small set of top-performing trials on the full training data and combines their
predictions into a single "ensemble" model. This can improve accuracy and reduce variance.

## Properties

| Property | Summary |
|:-----|:--------|
| `Enabled` | Gets or sets a value indicating whether AutoML should attempt to build an ensemble after the search. |
| `FinalSelectionPolicy` | Gets or sets the policy that determines whether the ensemble replaces the best single model. |
| `MaxModelCount` | Gets or sets the maximum number of top trials to include in the ensemble. |

