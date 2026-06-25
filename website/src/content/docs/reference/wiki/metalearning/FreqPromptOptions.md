---
title: "FreqPromptOptions<T, TInput, TOutput>"
description: "Configuration options for FreqPrompt: Frequency-domain prompt tuning for few-shot learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for FreqPrompt: Frequency-domain prompt tuning for few-shot learning.

## How It Works

FreqPrompt meta-learns additive parameter modulations ("prompts") in a frequency-domain
decomposition. Low-frequency prompts capture coarse domain shifts while high-frequency
prompts handle fine-grained task-specific adjustments. During adaptation, only the
prompt coefficients are updated, keeping the backbone frozen.

## Properties

| Property | Summary |
|:-----|:--------|
| `HighFreqPenalty` | Regularization weight for high-frequency prompt components (encourages smooth prompts). |
| `NumFreqComponents` | Number of frequency basis components for the prompt. |
| `PromptInitScale` | Scale of prompt initialization. |

