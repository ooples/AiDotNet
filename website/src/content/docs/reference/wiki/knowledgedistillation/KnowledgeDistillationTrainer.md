---
title: "KnowledgeDistillationTrainer<T>"
description: "Standard knowledge distillation trainer that uses a fixed teacher model to train a student."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Standard knowledge distillation trainer that uses a fixed teacher model to train a student.

## For Beginners

This is the standard implementation of knowledge distillation.
It takes a large, accurate teacher model and uses it to train a smaller, faster student model.

## How It Works

The training process works as follows:

1. For each input, get predictions from both teacher and student
2. Compute distillation loss (how different are their predictions?)
3. Update student parameters to minimize this loss
4. Repeat until student learns to mimic teacher

**Real-world Analogy:**
Think of this as an apprenticeship program. The master (teacher) demonstrates how to solve
problems, and the apprentice (student) learns by trying to replicate the master's approach.
The apprentice doesn't just learn the final answers, but also the reasoning process.

**Benefits of Knowledge Distillation:**

- **Model Compression**: Deploy a 10x smaller model with >90% of original accuracy
- **Faster Inference**: Smaller models run much faster on edge devices
- **Ensemble Distillation**: Combine knowledge from multiple teachers into one student
- **Transfer Learning**: Transfer knowledge across different architectures

**Success Stories:**

- DistilBERT: 40% smaller than BERT, 97% of performance, 60% faster
- MobileNet: Distilled from ResNet, 10x fewer parameters, deployable on phones
- TinyBERT: 7.5x smaller than BERT, suitable for edge deployment

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KnowledgeDistillationTrainer(ITeacherModel<Vector<>,Vector<>>,IDistillationStrategy<>,DistillationCheckpointConfig,Boolean,Double,Int32,Nullable<Int32>)` | Initializes a new instance of the KnowledgeDistillationTrainer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTeacherPredictions(Vector<>,Int32)` | Gets teacher predictions by calling the teacher model's GetLogits method. |

