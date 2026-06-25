---
title: "IAdaptedMetaModel<T>"
description: "Extended interface for meta-learning adapted models that carry task-specific adaptation state beyond backbone parameters."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Extended interface for meta-learning adapted models that carry task-specific adaptation state
beyond backbone parameters. Enables downstream code to access algorithm-specific adapted
representations for evaluation, analysis, or custom classification.

## For Beginners

When a meta-learner adapts to a new task, it often computes
enriched feature representations beyond just the backbone model's output. This interface
lets you access those enriched features for custom use cases, like building a nearest-neighbor
classifier on the adapted features instead of relying solely on the backbone's predictions.

## How It Works

Meta-learning algorithms like EPNet, MCL, SetFeat, and ConstellationNet compute adapted
representations during task adaptation (e.g., propagated features, projected features,
set-encoded class representations). This interface exposes those representations so that
evaluation code, visualization tools, or custom classifiers can use them.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptedSupportFeatures` | Gets the adapted support features computed during task adaptation. |
| `ParameterModulationFactors` | Gets the parameter modulation factors computed during adaptation. |

