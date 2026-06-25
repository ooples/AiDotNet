---
title: "OpenMAMLPlusOptions<T, TInput, TOutput>"
description: "Configuration options for Open-MAML++: MAML extended for open-set recognition with novelty detection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Open-MAML++: MAML extended for open-set recognition with
novelty detection.

## How It Works

Open-MAML++ extends MAML to handle open-set scenarios where test tasks may contain
classes not seen during meta-training. It meta-learns per-parameter learning rates
(like MAML++) and a novelty detection threshold based on prediction entropy. During
adaptation, predictions with entropy above the threshold are flagged as novel/unknown.

## Properties

| Property | Summary |
|:-----|:--------|
| `EntropyRegWeight` | Weight for the entropy regularization loss. |
| `InitialNoveltyThreshold` | Initial novelty threshold (entropy-based). |
| `LearnPerParamLR` | Whether to meta-learn per-parameter learning rates (MAML++ style). |
| `MultiStepLossWeight` | Multi-step loss coefficient: weight for intermediate adaptation step losses. |

