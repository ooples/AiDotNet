---
title: "RLAutoMLOptions<T>"
description: "Configuration options for running AutoML over reinforcement learning agents and hyperparameters."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for running AutoML over reinforcement learning agents and hyperparameters.

## For Beginners

RL AutoML tries a few different RL agent settings, measures which one earns
the most reward, and then trains the best configuration for your full training budget.

## How It Works

This options class is designed for use with `AiModelBuilder` when training RL agents.
It supports the AiDotNet facade pattern by providing sensible defaults while allowing customization.

## Properties

| Property | Summary |
|:-----|:--------|
| `CandidateAgents` | Gets or sets the allowed agent types for RL AutoML. |
| `EvaluationEpisodesPerTrial` | Gets or sets the number of evaluation episodes to run per AutoML trial (no learning). |
| `MaxStepsPerEpisodeOverride` | Gets or sets an optional maximum step count per episode override for AutoML trials. |
| `SearchSpaceOverrides` | Gets or sets optional hyperparameter search-space overrides. |
| `TrainingEpisodesPerTrial` | Gets or sets the number of training episodes to run per AutoML trial. |

