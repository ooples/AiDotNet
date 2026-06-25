---
title: "DiscoRLOptions<T, TInput, TOutput>"
description: "Configuration options for DiscoRL: Discovery-based meta-RL with skill discovery."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for DiscoRL: Discovery-based meta-RL with skill discovery.

## How It Works

DiscoRL discovers reusable "skills" (parameter subspaces) during meta-training
and combines them for new task adaptation. Each skill corresponds to a low-rank
direction in parameter space. A skill selector (gating network) chooses which
skills to activate for each task based on early gradient signals.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumSkills` | Number of discoverable skills. |
| `SelectionTemperature` | Temperature for skill selection softmax. |
| `SkillEntropyBonus` | Entropy bonus to encourage diverse skill usage. |
| `SkillRank` | Rank of each skill direction in parameter space. |

