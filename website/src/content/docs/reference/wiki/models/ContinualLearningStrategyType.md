---
title: "ContinualLearningStrategyType"
description: "Specifies the continual learning strategy to use for preventing catastrophic forgetting."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the continual learning strategy to use for preventing catastrophic forgetting.

## For Beginners

Continual learning strategies help neural networks learn new tasks
without forgetting previously learned ones. Different strategies use different approaches to
balance learning new knowledge while preserving old knowledge.

## Fields

| Field | Summary |
|:-----|:--------|
| `AveragedGEM` | Averaged GEM - efficient variant using single averaged constraint. |
| `EWC` | Elastic Weight Consolidation - penalizes changes to important weights using Fisher information. |
| `ExperienceReplay` | Experience Replay - stores and replays examples from previous tasks. |
| `GEM` | Gradient Episodic Memory - constrains gradients to not hurt stored examples. |
| `GenerativeReplay` | Generative Replay - uses generative model to create pseudo-examples for rehearsal. |
| `LearningWithoutForgetting` | Learning without Forgetting - uses knowledge distillation to preserve old predictions. |
| `MAS` | Memory Aware Synapses - unsupervised importance estimation using output sensitivity. |
| `OnlineEWC` | Online EWC - memory-efficient variant that maintains running Fisher estimate. |
| `PackNet` | PackNet - isolates parameters per task through pruning and freezing. |
| `ProgressiveNeuralNetworks` | Progressive Neural Networks - adds new columns with lateral connections for each task. |
| `SynapticIntelligence` | Synaptic Intelligence - tracks weight importance online during training. |
| `VCL` | Variational Continual Learning - Bayesian approach using posterior as prior. |

