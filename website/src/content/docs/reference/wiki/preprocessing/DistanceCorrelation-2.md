---
title: "DistanceCorrelation<T>"
description: "Distance Correlation for detecting nonlinear relationships between features and target."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Distance Correlation for detecting nonlinear relationships between features and target.

## For Beginners

Pearson correlation can miss relationships where one variable
depends on another in a curved or complex way. Distance correlation catches all types
of relationships. If it's zero, the variables are truly independent; if it's high,
there's some kind of relationship (linear or not).

## How It Works

Distance correlation measures both linear and nonlinear dependencies between variables.
Unlike Pearson correlation which only detects linear relationships, distance correlation
equals zero if and only if the variables are statistically independent.

