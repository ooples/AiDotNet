---
title: "AlignmentMethodOptions<T>"
description: "Configuration options for AI alignment methods."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for AI alignment methods.

## For Beginners

These settings control how your AI learns to behave according
to human values. You can adjust how much to weight human feedback, what principles to follow,
and how thoroughly to test for problems.

## How It Works

These options control how models are aligned with human values and intentions through
techniques like RLHF, constitutional AI, and red teaming.

## Properties

| Property | Summary |
|:-----|:--------|
| `CritiqueIterations` | Gets or sets the number of critique iterations for constitutional AI. |
| `EnableRedTeaming` | Gets or sets whether to perform red teaming. |
| `Gamma` | Gets or sets the discount factor for reward modeling. |
| `KLCoefficient` | Gets or sets the KL divergence penalty coefficient. |
| `LearningRate` | Gets or sets the learning rate for alignment training. |
| `RedTeamingAttempts` | Gets or sets the number of red teaming attempts. |
| `RewardModelArchitecture` | Gets or sets the reward model architecture. |
| `TrainingIterations` | Gets or sets the number of training iterations for RLHF. |
| `UseConstitutionalAI` | Gets or sets whether to use constitutional AI principles. |

