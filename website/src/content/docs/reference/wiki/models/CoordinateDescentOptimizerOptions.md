---
title: "CoordinateDescentOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Coordinate Descent optimization algorithm, which optimizes a function by solving for one variable at a time while holding others constant."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Coordinate Descent optimization algorithm, which optimizes a function
by solving for one variable at a time while holding others constant.

## For Beginners

Imagine you're trying to find the lowest point in a valley. Coordinate
Descent is like first walking only north/south until you can't go any lower, then switching to only
east/west, then back to north/south, and so on. By taking turns moving in different directions, you
can eventually reach the bottom of the valley. This approach can be simpler than trying to move in
all directions at once.

## How It Works

Coordinate Descent is an optimization technique that minimizes a function by updating one coordinate
(or variable) at a time, while keeping all other coordinates fixed. This approach can be effective
for problems where optimizing along individual dimensions is easier than optimizing all dimensions
simultaneously.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient estimation. |
| `LearningRateDecreaseRate` | Gets or sets the rate at which the learning rate decreases when performance worsens. |
| `LearningRateIncreaseRate` | Gets or sets the rate at which the learning rate increases when performance improves. |
| `MomentumDecreaseRate` | Gets or sets the rate at which momentum decreases when performance worsens. |
| `MomentumIncreaseRate` | Gets or sets the rate at which momentum increases when performance improves. |

