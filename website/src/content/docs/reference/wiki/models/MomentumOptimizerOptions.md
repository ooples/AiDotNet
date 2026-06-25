---
title: "MomentumOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Momentum Optimizer, which enhances gradient descent by adding a fraction of the previous update direction to the current update."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Momentum Optimizer, which enhances gradient descent by adding
a fraction of the previous update direction to the current update.

## For Beginners

The Momentum Optimizer is like adding a "memory" to the learning process,
which helps the algorithm learn faster and more effectively.

Imagine you're rolling a ball down a hilly landscape to find the lowest point:

- Standard gradient descent is like gently nudging the ball in the downhill direction at each point
- Momentum is like letting the ball build up speed as it rolls

This has several advantages:

- The ball can roll through small bumps and plateaus without getting stuck
- It builds up speed in consistent directions, moving faster toward the solution
- It can dampen the "zig-zagging" that happens on steep slopes

This class lets you configure how the ball rolls: how fast it can go (learning rate),
how much momentum it builds up, and how these values adjust during training based on 
whether progress is being made.

## How It Works

The Momentum Optimizer is an extension of gradient descent that helps accelerate convergence and
reduce oscillation in the optimization process. It achieves this by accumulating a velocity vector
in the direction of persistent reduction in the objective function across iterations. This approach
allows the optimizer to build up "momentum" in consistent directions, helping it navigate flat regions
more quickly and dampening oscillations in directions with high curvature. Both the learning rate and
momentum coefficient can be adapted during training based on the optimization performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which the learning rate is decreased when the loss is getting worse. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which the learning rate is increased when the loss is improving. |
| `MaxLearningRate` | Gets or sets the maximum allowed learning rate for the optimization process. |
| `MomentumDecreaseFactor` | Gets or sets the factor by which the momentum coefficient is decreased when the loss is getting worse. |
| `MomentumIncreaseFactor` | Gets or sets the factor by which the momentum coefficient is increased when the loss is improving. |

