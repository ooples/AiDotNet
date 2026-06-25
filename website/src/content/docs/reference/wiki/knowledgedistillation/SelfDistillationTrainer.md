---
title: "SelfDistillationTrainer<T>"
description: "Implements self-distillation where a model acts as its own teacher to improve calibration and generalization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Implements self-distillation where a model acts as its own teacher to improve calibration and generalization.

## For Beginners

Self-distillation is a clever technique where a model learns from itself!
Instead of using a separate larger teacher, you train a model normally, then use it as a teacher
to train itself again. This often improves:

- **Calibration**: Model confidence matches actual accuracy
- **Generalization**: Better performance on unseen data
- **Robustness**: Less sensitive to noisy labels or adversarial examples

## How It Works

**How It Works:**

1. Train model normally on hard labels (standard training)
2. Save the trained model's predictions
3. Retrain the model using its own soft predictions as teacher
4. Repeat for multiple generations if desired

**Real-world Analogy:**
Imagine studying for an exam, then teaching the material to yourself as if you were a student.
By explaining concepts in your own words, you deepen your understanding and identify gaps
in your knowledge. Self-distillation works similarly for neural networks.

**Variants:**

- **Iterative Self-Distillation**: Multiple rounds of self-teaching
- **Born-Again Networks**: Same architecture, trained from scratch with self as teacher
- **Online Self-Distillation**: Student learns from earlier checkpoints of itself

**Benefits:**

- No need for a separate teacher model
- Improves calibration without model compression
- Can be combined with data augmentation for better regularization
- Often provides 1-3% accuracy improvement for free

**When to Use:**

- You want better calibrated predictions
- You have limited model capacity (can't afford a larger teacher)
- You want to improve an existing trained model
- You're training on noisy or imperfect labels

**References:**

- Furlanello, T., et al. (2018). Born Again Neural Networks. ICML.
- Zhang, L., et al. (2019). Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self-Distillation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfDistillationTrainer(IDistillationStrategy<>,Int32,Nullable<Int32>)` | Initializes a new instance of the SelfDistillationTrainer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EMADecay` | Gets or sets the EMA decay rate (default 0.99). |
| `UseEMA` | Gets or sets whether to use exponential moving average for teacher predictions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetTeacherPredictions(Vector<>,Int32)` | Gets teacher predictions from the cached predictions dictionary (for self-distillation). |
| `TrainMultipleGenerations(Func<Vector<>,Vector<>>,Action<Vector<>>,Vector<Vector<>>,Vector<Vector<>>,Int32,Int32,Action<Int32,>)` | Performs self-distillation training for the specified number of generations. |

