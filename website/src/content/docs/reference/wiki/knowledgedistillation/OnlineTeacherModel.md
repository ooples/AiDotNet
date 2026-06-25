---
title: "OnlineTeacherModel<T>"
description: "Online teacher model that updates its parameters during student training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Online teacher model that updates its parameters during student training.

## For Beginners

Unlike standard distillation where the teacher is frozen,
online distillation allows the teacher to continue learning during student training.
This is useful for:

- Continuous learning scenarios
- Evolving data distributions
- Co-training teacher and student simultaneously

## How It Works

**How It Works:**

1. Initialize teacher model (can be pre-trained or random)
2. During student training, also update teacher with new data
3. Teacher provides evolving knowledge to student
4. Both models improve together

**Real-world Analogy:**
Imagine a mentor and apprentice both continuing to learn as they work together.
The mentor (teacher) doesn't just transfer old knowledge - they also learn from new
experiences and share those insights with the apprentice (student).

**Use Cases:**

- **Streaming Data**: New data arrives continuously
- **Domain Adaptation**: Distribution shifts over time
- **Co-training**: Teacher and student help each other
- **Incremental Learning**: Models must adapt to new classes/tasks

**Update Strategies:**

- **EMA (Exponential Moving Average)**: Smooth updates, stable teacher
- **Periodic Sync**: Update teacher every N steps
- **Gradient-based**: Teacher trained with separate loss
- **Momentum**: Teacher follows student with momentum

**Advantages:**

- Adapts to changing data
- No need for pre-trained teacher
- Can improve teacher and student together
- Suitable for lifelong learning

**Challenges:**

- Risk of teacher forgetting/degrading
- Need careful update rate tuning
- More complex training dynamics
- Harder to debug

**References:**

- Zhang et al. (2018). Deep Mutual Learning. CVPR.
- Anil et al. (2018). Large Scale Distributed Neural Network Training through Online Distillation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnlineTeacherModel(Func<Vector<>,Vector<>>,Int32,Int32,Action<Vector<>,Vector<>>,OnlineUpdateMode,Double,Int32)` | Initializes a new instance of the OnlineTeacherModel class using function delegates. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsUpdating` | Gets or sets whether the teacher is currently updating. |
| `OutputDimension` | Gets the output dimension of the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits(Vector<>)` | Gets logits from the teacher model. |
| `PauseUpdates` | Pauses teacher updates (freezes teacher). |
| `ResetCounter` | Resets the update counter. |
| `ResumeUpdates` | Resumes teacher updates. |
| `Update(Vector<>,Vector<>)` | Updates the teacher model with new data. |
| `UpdateEMA(Vector<>,Vector<>)` | Updates teacher using exponential moving average. |
| `UpdateGradient(Vector<>,Vector<>)` | Updates teacher using gradient-based learning. |
| `UpdateMomentum(Vector<>,Vector<>)` | Updates teacher using momentum. |

