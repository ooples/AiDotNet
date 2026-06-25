---
title: "DistillationStrategyType"
description: "Specifies the type of knowledge distillation strategy to use for transferring knowledge from teacher to student models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the type of knowledge distillation strategy to use for transferring knowledge
from teacher to student models.

## For Beginners

Different distillation strategies focus on different aspects of
the teacher's knowledge. Some match final outputs, others match intermediate features or
relationships between samples.

## How It Works

**Choosing a Strategy:**

- Use **ResponseBased** for most cases (standard Hinton distillation)
- Use **FeatureBased** when student architecture differs significantly from teacher
- Use **AttentionBased** for transformer models (BERT, GPT)
- Use **RelationBased** to preserve relationships between samples
- Use **Contrastive** for self-supervised learning scenarios

## Fields

| Field | Summary |
|:-----|:--------|
| `ContrastiveBased` | Contrastive Representation Distillation / CRD (Tian et al., 2020). |
| `FactorTransfer` | Factor Transfer (Kim et al., 2018). |
| `FeatureBased` | Feature-based distillation / FitNets (Romero et al., 2014). |
| `FlowBased` | Flow of Solution Procedure / FSP (Yim et al., 2017). |
| `Hybrid` | Combined/Hybrid distillation. |
| `RelationBased` | Relational Knowledge Distillation / RKD (Park et al., 2019). |
| `ResponseBased` | Response-based distillation (Hinton et al., 2015). |
| `SelfDistillation` | Self-distillation (Zhang et al., 2019; Furlanello et al., 2018). |
| `VariationalInformation` | Variational Information Distillation / VID (Ahn et al., 2019). |

