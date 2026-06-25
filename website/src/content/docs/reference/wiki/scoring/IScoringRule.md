---
title: "IScoringRule<T>"
description: "Defines a proper scoring rule for evaluating probabilistic predictions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Scoring`

Defines a proper scoring rule for evaluating probabilistic predictions.

## For Beginners

When you make a probabilistic prediction (like "70% chance of rain"),
scoring rules tell you how good your prediction was after you see what actually happened.
A proper scoring rule rewards you most for predicting probabilities that match reality -
you can't game the system by being overconfident or underconfident.

## How It Works

A scoring rule measures how well a predicted probability distribution matches
the observed outcomes. Proper scoring rules are maximized (or minimized) when
the predicted distribution equals the true distribution, incentivizing honest forecasts.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsMinimized` | Gets whether this scoring rule should be minimized (true) or maximized (false). |
| `Name` | Gets the name of this scoring rule. |

## Methods

| Method | Summary |
|:-----|:--------|
| `MeanScore(IParametricDistribution<>[],Vector<>)` | Computes the mean score over multiple prediction-observation pairs. |
| `Score(IParametricDistribution<>,)` | Computes the score for a single prediction-observation pair. |
| `ScoreGradient(IParametricDistribution<>,)` | Computes the gradient of the score with respect to distribution parameters. |

