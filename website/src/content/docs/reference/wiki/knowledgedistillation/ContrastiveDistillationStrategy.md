---
title: "ContrastiveDistillationStrategy<T>"
description: "Implements Contrastive Representation Distillation (CRD) which transfers knowledge through contrastive learning of sample relationships rather than just matching outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Implements Contrastive Representation Distillation (CRD) which transfers knowledge through
contrastive learning of sample relationships rather than just matching outputs.

## For Beginners

Contrastive distillation teaches the student to understand which
samples are similar and which are different, not just to copy the teacher's predictions.
It's like learning to group things by their similarities rather than just memorizing labels.

## How It Works

**Real-world Analogy:**
Instead of just teaching a student "This is a dog," you teach them "Dogs are more similar
to wolves than to cats" and "Retrievers are more similar to Labs than to Chihuahuas."
This relational understanding helps the student generalize better to new examples.

**How CRD Works:**

1. Extract embeddings/features from teacher and student
2. For each sample (anchor), identify:
- Positive samples: Same class or similar features
- Negative samples: Different class or dissimilar features
3. Pull student embeddings of anchor and positives together
4. Push student embeddings of anchor and negatives apart
5. Ensure student's embedding space has same structure as teacher's

**Key Differences from Standard Distillation:**

- **Standard**: Match output probabilities [0.1, 0.7, 0.2]
- **CRD**: Match embedding similarities and distances
- **Benefit**: Better generalization, especially for few-shot learning

**Mathematical Foundation:**
CRD uses InfoNCE loss (Noise Contrastive Estimation):
L = -log(exp(sim(t_i, s_i)/τ) / Σ_j exp(sim(t_i, s_j)/τ))
where:

- t_i, s_i are teacher/student embeddings of sample i
- τ is temperature
- j ranges over all samples in batch

**Benefits:**

- **Better Features**: Student learns richer representations
- **Few-Shot Learning**: Transfers better to new classes
- **Robustness**: Less sensitive to label noise
- **Interpretability**: Embedding space is more structured
- **Complementary**: Can combine with standard distillation

**Use Cases:**

- Few-shot/zero-shot learning
- Transfer learning across domains
- Learning with noisy labels
- Metric learning tasks (face recognition, image retrieval)
- Self-supervised pre-training

**Performance Improvements:**

- CRD often gives 2-4% better accuracy than standard distillation
- Particularly strong for small student models
- Excellent for tasks requiring good embeddings

**References:**

- Tian et al. (2020). Contrastive Representation Distillation. ICLR.
- Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContrastiveDistillationStrategy(String,Double,Double,Double,Int32,ContrastiveMode)` | Initializes a new instance of the ContrastiveDistillationStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeContrastiveLoss(Vector<>[],Vector<>[],Int32[])` | Computes contrastive loss on embeddings/features. |
| `ComputeCosineSimilarityGradient(Vector<>,Vector<>)` | Computes gradient of cosine similarity with respect to the first vector. |
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes gradient of standard loss. |
| `ComputeInfoNCELoss(Vector<>[],Vector<>[],Int32[])` | Computes InfoNCE (Noise Contrastive Estimation) loss. |
| `ComputeIntermediateGradient(IntermediateActivations<>,IntermediateActivations<>)` | Computes gradients of intermediate activation loss with respect to student embeddings. |
| `ComputeIntermediateLoss(IntermediateActivations<>,IntermediateActivations<>)` | Computes intermediate activation loss by matching embedding space structure between teacher and student. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes standard output loss (contrastive loss computed separately on embeddings). |
| `ComputeNTXentGradient(Vector<>[],Vector<>[],Int32)` | Computes NT-Xent gradient for a single sample. |
| `ComputeNTXentLoss(Vector<>[],Vector<>[])` | Computes NT-Xent (Normalized Temperature-scaled Cross Entropy) loss. |
| `ComputeTripletLoss(Vector<>[],Vector<>[],Int32[])` | Computes triplet loss. |

