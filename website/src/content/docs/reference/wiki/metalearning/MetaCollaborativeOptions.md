---
title: "MetaCollaborativeOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-Collaborative Learning across multiple task domains."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-Collaborative Learning across multiple task domains.

## How It Works

Meta-Collaborative Learning uses gradient alignment between concurrently adapted tasks
to transfer cross-domain knowledge. Tasks with aligned gradient directions reinforce each
other, while conflicting gradients are dampened. A domain-specific momentum buffer per task
stabilizes cross-task transfer.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignmentWeight` | Weight for the gradient alignment loss between tasks. |
| `GradientMomentum` | Momentum coefficient for the domain-specific gradient buffer. |
| `NumDomainSlots` | Number of domain slots for maintaining separate gradient histories. |

