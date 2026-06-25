---
title: "PolicyDistributionHelper<T>"
description: "Provides tape-differentiable policy distribution computations for reinforcement learning."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides tape-differentiable policy distribution computations for reinforcement learning.
All methods are engine-routed so gradients flow through the gradient tape; the calling
agent passes its own `IEngine` so the helper never reaches for a static
`AiDotNetEngine.Current` (which would break per-call engine overrides, DI in
serving / dashboard scenarios, and test-time engine isolation).

## How It Works

**For Beginners:** In reinforcement learning, the agent's "policy" is a probability
distribution over actions. To train the policy, we need to compute:

- **Log-probability:** How likely was the action the agent took? (used in policy gradient)
- **Entropy:** How "spread out" is the distribution? (used to encourage exploration)

These computations must use engine operations (not scalar math) so that the gradient
tape can automatically compute how to adjust the policy network's weights.

**Discrete actions:** The network outputs logits (raw scores) for each action.
We apply softmax to get probabilities, then take log of the selected action's probability.

**Continuous actions:** The network outputs mean and log-standard-deviation for a
Gaussian (normal) distribution. The log-probability follows the Gaussian formula.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDiscreteEntropy(IEngine,Tensor<>)` | Computes entropy of a discrete distribution from logits using engine ops. |
| `ComputeDiscreteLogProb(IEngine,Tensor<>,Int32[])` | Computes log-probabilities for discrete actions from logits using engine ops. |
| `ComputeGaussianEntropy(IEngine,Tensor<>)` | Computes entropy of a Gaussian distribution using engine ops. |
| `ComputeGaussianLogProb(IEngine,Tensor<>,Tensor<>,Tensor<>)` | Computes log-probabilities for continuous (Gaussian) actions using engine ops. |

