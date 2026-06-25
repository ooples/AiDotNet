---
title: "RobustnessStats<T>"
description: "Represents adversarial robustness diagnostics aggregated over a dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents adversarial robustness diagnostics aggregated over a dataset.

## For Beginners

This stores summary robustness metrics (like accuracy under attack)
for an entire dataset, helping you understand how well your model resists adversarial perturbations.

Key concepts:

- **Clean Accuracy**: How accurate the model is on unmodified inputs
- **Adversarial Accuracy**: How accurate the model is when inputs are perturbed by attacks
- **Certified Accuracy**: The fraction of samples with provably correct predictions within a perturbation radius
- **Attack Success Rate**: How often an attacker can fool the model
- **Average Perturbation Size**: How much inputs need to be changed to fool the model

## How It Works

This container is designed to integrate with the existing AiDotNet evaluation pipeline by living alongside
`ErrorStats` and `PredictionStats` within `DataSetStats`.
It stores metrics related to model robustness against adversarial attacks and certified defenses.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdditionalMetrics` | Gets a dictionary of additional robustness metrics. |
| `AdversarialAccuracy` | Gets the accuracy of the model on adversarially perturbed inputs. |
| `AttackSuccessRate` | Gets the fraction of inputs for which the attack successfully fooled the model. |
| `AttackType` | Gets the type of attack used for adversarial robustness evaluation. |
| `AverageCertifiedRadius` | Gets the average certified robustness radius across samples. |
| `AveragePerturbationSize` | Gets the average size of perturbations needed to create successful adversarial examples. |
| `CertifiedAccuracy` | Gets the certified accuracy at the specified perturbation radius. |
| `CleanAccuracy` | Gets the accuracy of the model on clean (unperturbed) inputs. |
| `EvaluationEpsilon` | Gets the perturbation radius (epsilon) used for robustness evaluation. |
| `IsEvaluated` | Gets or sets whether robustness evaluation has been performed. |
| `NormType` | Gets the norm type used for measuring perturbation size (e.g., "L2", "Linf"). |
| `RobustnessScore` | Gets a combined robustness score (0-1) that balances clean and adversarial performance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Empty` | Creates an empty `RobustnessStats` instance. |

