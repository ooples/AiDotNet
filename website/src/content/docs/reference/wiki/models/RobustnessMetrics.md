---
title: "RobustnessMetrics<T>"
description: "Contains metrics for evaluating adversarial robustness of models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Contains metrics for evaluating adversarial robustness of models.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdditionalMetrics` | Gets or sets additional evaluation metrics. |
| `AdversarialAccuracy` | Gets or sets the accuracy on adversarial examples. |
| `AttackSuccessRate` | Gets or sets the attack success rate (percentage of successful attacks). |
| `AveragePerturbationSize` | Gets or sets the average perturbation size needed to fool the model. |
| `CleanAccuracy` | Gets or sets the accuracy on clean (non-adversarial) examples. |
| `RobustnessScore` | Gets or sets the robustness score (combines clean and adversarial accuracy). |

