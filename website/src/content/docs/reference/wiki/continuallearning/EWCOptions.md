---
title: "EWCOptions<T>"
description: "Configuration options for Elastic Weight Consolidation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Strategies`

Configuration options for Elastic Weight Consolidation.

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambda` | Gets or sets the regularization strength (lambda). |
| `MinFisherValue` | Gets or sets the minimum Fisher Information value (to prevent division by zero). |
| `NormalizeFisher` | Gets or sets whether to normalize Fisher Information values. |
| `NumFisherSamples` | Gets or sets the number of samples for Fisher Information estimation. |
| `OnlineDecayFactor` | Gets or sets the decay factor for online EWC (gamma in the paper). |
| `UseOnlineEWC` | Gets or sets whether to use online EWC (accumulate Fisher info across tasks). |

