---
title: "SentenceTransformersFineTuner<T>"
description: "Fine-tuner for sentence transformer embedding models on domain-specific training data using triplet loss."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels`

Fine-tuner for sentence transformer embedding models on domain-specific training data using triplet loss.

## For Beginners

Think of fine-tuning like teaching a translator specialized vocabulary.

Pre-trained model (general knowledge):

- "bank" → embedding that works for both "river bank" and "financial bank"
- Problem: Not precise for your specific domain!

Fine-tuned model (specialized):

- If you're building a financial app, train it with examples:
- Anchor: "bank account"
- Positive: "savings account" (similar in YOUR domain)
- Negative: "river bank" (different in YOUR domain)
- Result: Model learns "bank" means "financial institution" in your context

Real-world example:
Medical domain:

- General model: "cold" could mean temperature or illness
- After fine-tuning with medical data:
- Anchor: "patient has cold"
- Positive: "patient has flu" (similar symptoms)
- Negative: "cold weather" (unrelated)
- Model now correctly groups medical conditions together

## How It Works

Fine-tuning adapts pre-trained embedding models to perform better on specific domains or tasks by
training on custom (anchor, positive, negative) triplets. This improves embedding quality for
specialized use cases like legal documents, medical terminology, or company-specific content.

**Example Usage:**

**How It Works:**
Training process:

1. Triplet Loss Function:
- Anchor embedding (A): Embed("fraud detection")
- Positive embedding (P): Embed("fraudulent transaction") - should be similar
- Negative embedding (N): Embed("legitimate payment") - should be different
- Loss = max(0, distance(A,P) - distance(A,N) + margin)
- Goal: Make distance(A,P) small and distance(A,N) large

2. Training Loop:
- For each epoch (10 iterations):
* For each training triplet:
- Generate embeddings for anchor, positive, negative
- Calculate triplet loss
- Update model weights to minimize loss
- Cache updated embeddings

3. Result:
- Model learns to embed domain-specific texts closer together
- Generic texts pushed further apart
- Improved retrieval accuracy for your specific use case

Current implementation simulates training with embedding caching.
Real training requires gradient descent and backpropagation through the neural network.

**Benefits:**

- Domain adaptation - Customize embeddings for specific industry/task
- Improved accuracy - Better retrieval performance on your data
- Less training data - Fine-tuning needs 100-10,000 examples vs millions for pre-training
- Transfer learning - Leverages existing knowledge from pre-trained model
- Cost-effective - Faster and cheaper than training from scratch

**Limitations:**

- Requires quality training data (good triplets are crucial)
- Can overfit with too few examples (aim for 1,000+ triplets minimum)
- Needs domain expertise to create meaningful triplets
- Current implementation simulates training (real training requires ML framework)
- Training time increases with model size and dataset size

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SentenceTransformersFineTuner(String,String,Int32,,Int32)` | Initializes a new instance of the `SentenceTransformersFineTuner` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseModel` | Gets the base ONNX model, loading it lazily on first access. |
| `EmbeddingDimension` |  |
| `MaxTokens` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EmbedCore(String)` |  |
| `FineTune(IEnumerable<ValueTuple<String,String,String>>)` | Fine-tunes the model on provided training data. |

