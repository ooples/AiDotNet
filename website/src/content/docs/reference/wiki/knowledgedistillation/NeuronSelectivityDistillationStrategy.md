---
title: "NeuronSelectivityDistillationStrategy<T>"
description: "Neuron selectivity distillation that transfers the activation patterns and selectivity of individual neurons."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Neuron selectivity distillation that transfers the activation patterns and selectivity of individual neurons.

## How It Works

**For Production Use:** This strategy focuses on matching how individual neurons respond to inputs.
Some neurons are highly selective (activate strongly for specific patterns), while others are more general.
Transferring this selectivity helps the student learn meaningful feature representations.

**Key Concept:** Neuron selectivity measures how discriminative each neuron is. A highly selective
neuron activates strongly for certain inputs and weakly for others. The distribution of selectivity across
neurons is important for model performance.

**Implementation:** We measure selectivity using:

1. Activation variance (how much neuron output varies across samples)
2. Sparsity (what percentage of time the neuron is active)
3. Peak-to-average ratio (how peaked the activation distribution is)

**Usage Pattern:** This strategy implements both standard output-based distillation and
intermediate activation-based selectivity matching. Use as follows:

**Standard Usage (via IDistillationStrategy):**

**With Intermediate Activations (via IIntermediateActivationStrategy):**

The selectivityWeight and metric parameters control the intermediate activation loss component.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes the gradient of the base distillation loss on final outputs. |
| `ComputeIntermediateGradient(IntermediateActivations<>,IntermediateActivations<>)` | Computes gradients of intermediate activation loss with respect to student activations. |
| `ComputeIntermediateLoss(IntermediateActivations<>,IntermediateActivations<>)` | Computes intermediate activation loss by matching neuron selectivity between teacher and student. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes the base distillation loss on final outputs. |
| `ComputePeakToAverageGradient(Vector<>[],Double[],Double[],Matrix<>)` | Computes gradient for peak-to-average ratio selectivity loss. |
| `ComputeSelectivityLoss(Vector<>[],Vector<>[])` | Computes neuron selectivity loss by comparing activation patterns across a batch. |
| `ComputeSparsityGradient(Vector<>[],Double[],Double[],Matrix<>)` | Computes gradient for sparsity-based selectivity loss. |
| `ComputeVarianceGradient(Vector<>[],Double[],Double[],Matrix<>)` | Computes gradient for variance-based selectivity loss. |

