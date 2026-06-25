---
title: "AttentionDistillationStrategy<T>"
description: "Implements attention-based knowledge distillation for transformer models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Implements attention-based knowledge distillation for transformer models.
Transfers knowledge through attention patterns rather than just final outputs.

## For Beginners

Attention mechanisms in transformers tell us "what the model is focusing on."
Instead of just copying the teacher's final answers, attention distillation teaches the student to
focus on the same things the teacher focuses on.

## How It Works

**Real-world Analogy:**
Imagine learning to play chess from a grandmaster. Instead of just copying their moves (outputs),
you also learn where they look on the board and what pieces they pay attention to. This deeper
understanding helps you think like the master, not just mimic their moves.

**Why Attention Distillation?**

- **Richer Knowledge**: Attention patterns reveal reasoning process
- **Better for Transformers**: Transformers rely heavily on attention
- **Interpretability**: Can see what student learned to focus on
- **Complementary**: Works with response-based distillation

**How It Works:**

1. Extract attention weights from teacher layers
2. Extract attention weights from student layers
3. Minimize MSE between attention distributions
4. Combine with standard output distillation loss

**Attention Matching Strategies:**

- **Layer-wise**: Match corresponding layers (layer 6→layer 3)
- **Head-wise**: Match individual attention heads
- **Global**: Match averaged attention across all heads
- **Selective**: Match only the most important heads

**Common Applications:**

- **DistilBERT**: Used attention distillation to compress BERT
- **TinyBERT**: Attention transfer + representation transfer
- **MobileBERT**: Layer-wise attention matching
- **Vision Transformers**: Attention distillation for ViT compression

**Benefits:**

- Preserves model's "reasoning" process
- Improves student's interpretability
- Often yields 2-5% better accuracy than output-only distillation
- Helps with few-shot and zero-shot transfer

**References:**

- Sanh et al. (2019). DistilBERT: A Distilled Version of BERT. arXiv:1910.01108
- Jiao et al. (2020). TinyBERT: Distilling BERT for Natural Language Understanding. EMNLP.
- Wang et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AttentionDistillationStrategy(String[],Double,Double,Double,AttentionMatchingMode)` | Initializes a new instance of the AttentionDistillationStrategy class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAttentionLoss(Func<String,Vector<>>,Func<String,Vector<>>)` | Computes attention matching loss between teacher and student attention patterns. |
| `ComputeAttentionMatchingGradient(Vector<>,Vector<>)` | Computes gradient of attention matching loss for a single sample. |
| `ComputeAttentionMatchingLoss(Vector<>,Vector<>)` | Computes loss for matching a single attention layer. |
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes gradient of the combined loss. |
| `ComputeIntermediateGradient(IntermediateActivations<>,IntermediateActivations<>)` | Computes gradients of intermediate activation loss with respect to student activations. |
| `ComputeIntermediateLoss(IntermediateActivations<>,IntermediateActivations<>)` | Computes intermediate activation loss by matching attention patterns between teacher and student. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes combined distillation loss (output loss + attention loss). |

