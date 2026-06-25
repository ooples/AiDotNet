---
title: "TradingAgentOptions<T>"
description: "Configuration options for financial trading agents."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for financial trading agents.

## For Beginners

These options control how the trading agent behaves:

**RL Parameters:**

- LearningRate: How fast the agent learns (higher = faster but unstable)
- DiscountFactor: How much to value future rewards (higher = more patient)
- BatchSize: How many experiences to learn from at once
- ReplayBufferSize: How many past experiences to remember

**Trading Parameters:**

- InitialCapital: Starting money for the portfolio
- TransactionCost: Cost per trade (brokers charge fees)
- MaxPositionSize: Maximum allowed position (prevents over-concentration)
- RiskFreeRate: Benchmark return (usually government bond rate)

**Risk Management:**

- UseRiskAdjustedReward: Whether to penalize high variance
- VariancePenalty: How much to penalize risky behavior

## How It Works

TradingAgentOptions extends standard RL options with financial-specific parameters
for trading environments, risk management, and reward calculation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSize` | Number of possible actions or continuous action dimensions. |
| `AllowShortSelling` | Whether to allow short selling. |
| `AutoTuneAlpha` | Whether to automatically tune SAC temperature. |
| `BatchSize` | Batch size for training updates. |
| `ContinuousActions` | Whether the action space is continuous. |
| `DiscountFactor` | Discount factor (gamma) for future rewards. |
| `EntropyCoefficient` | Entropy coefficient for exploration. |
| `EpsilonDecay` | Exploration decay rate. |
| `EpsilonEnd` | Final exploration rate. |
| `EpsilonStart` | Initial exploration rate. |
| `GAELambda` | GAE lambda for advantage estimation. |
| `HiddenLayers` | Hidden layer sizes for the neural network. |
| `InitialCapital` | Initial capital for the trading portfolio. |
| `LearningRate` | Learning rate for gradient updates. |
| `LossFunction` | Loss function for training the agent's neural networks. |
| `MaxPositionSize` | Maximum position size as a fraction of portfolio. |
| `PPOClipRange` | Clip range for PPO algorithm. |
| `ReplayBufferSize` | Size of the experience replay buffer. |
| `RewardScale` | Reward scaling factor. |
| `RiskFreeRate` | Annual risk-free rate for Sharpe ratio calculation. |
| `SACAlpha` | Temperature parameter for SAC algorithm. |
| `Seed` | Random seed for reproducibility. |
| `StateSize` | Dimension of the market state observation. |
| `TargetUpdateFrequency` | How often to update the target network. |
| `Tau` | Soft update coefficient for target networks. |
| `TransactionCost` | Transaction cost as a fraction of trade value. |
| `UseRiskAdjustedReward` | Whether to use risk-adjusted rewards for training. |
| `ValueCoefficient` | Value function coefficient for actor-critic methods. |
| `VariancePenalty` | Penalty coefficient for return variance. |
| `WarmupSteps` | Number of steps before training begins. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and throws if invalid. |

