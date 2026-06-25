---
title: "NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Nesterov Accelerated Gradient optimization algorithm, a momentum-based technique that improves convergence speed in gradient descent optimization."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Nesterov Accelerated Gradient optimization algorithm, a momentum-based
technique that improves convergence speed in gradient descent optimization.

## For Beginners

Nesterov Accelerated Gradient is a technique that helps AI models learn faster and better.

Imagine you're trying to find the lowest point in a valley by walking downhill:

- Regular gradient descent is like always taking a step directly downhill from where you stand
- Adding momentum is like rolling a ball downhill - it picks up speed and can go faster
- Nesterov adds a clever twist: it looks ahead in the direction the ball is rolling before deciding which way is downhill

This "look-ahead" approach helps the model:

- Learn faster in most situations
- Avoid overshooting the best solution
- Navigate tricky terrain in the learning landscape
- Adapt to different types of problems

The settings in this class let you control:

- How quickly the learning rate and momentum can increase when things are going well
- How quickly they decrease when progress slows down

This adaptive behavior helps the model automatically find efficient settings as it learns,
rather than requiring you to find the perfect fixed values upfront.

## How It Works

Nesterov Accelerated Gradient (NAG) is an enhancement to standard gradient descent optimization that 
incorporates momentum with a look-ahead approach. By evaluating gradients at a position estimated 
by the momentum term rather than the current position, NAG provides better responsiveness to changes 
in the error surface. This results in faster convergence rates and improved performance, particularly 
in problems with high curvature or narrow valleys in the error surface. The algorithm adaptively 
adjusts both learning rate and momentum during training to optimize performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which the learning rate is decreased when the algorithm is not making good progress. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which the learning rate is increased when the algorithm is making good progress. |
| `MomentumDecreaseFactor` | Gets or sets the factor by which the momentum is decreased when the algorithm is not making good progress. |
| `MomentumIncreaseFactor` | Gets or sets the factor by which the momentum is increased when the algorithm is making good progress. |

