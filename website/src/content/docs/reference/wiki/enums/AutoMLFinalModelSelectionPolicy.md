---
title: "AutoMLFinalModelSelectionPolicy"
description: "Defines how AutoML chooses the final model to return after the search completes."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines how AutoML chooses the final model to return after the search completes.

## For Beginners

An ensemble combines multiple models to often improve accuracy and stability.
This setting controls whether AutoML should return the best single model or an ensemble.

## How It Works

AutoML can return the single best trial, or it can optionally build an ensemble from top trials
and return that instead.

## Fields

| Field | Summary |
|:-----|:--------|
| `AlwaysUseEnsemble` | Always return an ensemble when enough successful trials exist. |
| `BestSingleModel` | Always return the best single trial model. |
| `UseEnsembleIfBetter` | Build an ensemble and return it if it scores better than the best single model. |

