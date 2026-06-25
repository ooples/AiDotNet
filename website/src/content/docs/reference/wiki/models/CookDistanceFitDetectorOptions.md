---
title: "CookDistanceFitDetectorOptions"
description: "Configuration options for the Cook's Distance fit detector, which helps identify influential data points and detect potential overfitting or underfitting in regression models."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Cook's Distance fit detector, which helps identify influential data points
and detect potential overfitting or underfitting in regression models.

## For Beginners

Imagine you're trying to draw a straight line through a set of points. Some points
might have a big effect on where you place that line - if you removed them, the line would move significantly.
These are called "influential points." Cook's Distance measures how influential each point is. This detector
looks at all your data points and checks if too many (or too few) are highly influential, which could indicate
problems with how well your model fits the data. If many points are influential, your model might be too simple
(underfitting). If very few points are influential, your model might be too complex (overfitting).

## How It Works

Cook's Distance is a statistical measure that identifies how much influence each data point has on a regression model.
Points with high Cook's Distance values have a disproportionate effect on the model's predictions and parameters.
This detector analyzes the distribution of Cook's Distance values across your dataset to identify potential
fitting problems.

## Properties

| Property | Summary |
|:-----|:--------|
| `InfluentialThreshold` | Gets or sets the threshold for determining when a data point is considered influential. |
| `OverfitThreshold` | Gets or sets the threshold for the proportion of influential points that suggests overfitting. |
| `UnderfitThreshold` | Gets or sets the threshold for the proportion of influential points that suggests underfitting. |

