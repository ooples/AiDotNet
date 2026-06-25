---
title: "GCDPLNetOptions<T, TInput, TOutput>"
description: "Configuration options for GCDPLNet (Graph-based Cross-Domain Prototype Learning Network)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for GCDPLNet (Graph-based Cross-Domain Prototype Learning Network).

## How It Works

GCDPLNet uses graph-based message passing between parameter groups to propagate
cross-domain knowledge. Parameter groups are treated as graph nodes, with learned
attention edges determining information flow during adaptation.

## Properties

| Property | Summary |
|:-----|:--------|
| `GraphAttentionDim` | Dimension of graph attention features. |
| `MessagePassingSteps` | Rounds of message passing between graph nodes. |
| `MessageWeight` | Weight on the message-passing-influenced adaptation. |
| `NumGraphNodes` | Number of parameter groups (graph nodes). |

