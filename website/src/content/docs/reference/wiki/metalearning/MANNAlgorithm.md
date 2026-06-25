---
title: "MANNAlgorithm<T, TInput, TOutput>"
description: "Implementation of Memory-Augmented Neural Networks (MANN) for meta-learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Memory-Augmented Neural Networks (MANN) for meta-learning.

## For Beginners

MANN uses an external memory like a human's working memory:

**How it works:**

1. Neural network processes input and produces a key
2. Uses key to read relevant information from external memory
3. Combines input with memory content to make prediction
4. During learning, writes new information to memory
5. Memory persists across episodes for lifelong learning

**Analogy:** Like a student with a notebook who can:

- Look up previous examples (reading memory)
- Write down new important information (writing memory)
- Use both to solve new problems

## How It Works

Memory-Augmented Neural Networks combine a neural network with an external memory
matrix. The network can read from and write to this memory, enabling rapid
learning by storing new information directly in memory during adaptation.

**Algorithm - Memory-Augmented Neural Networks:**

**Key Insights:**

1. **Rapid Learning**: New information can be stored in memory with a single

write operation, enabling one-shot learning.

2. **Continuous Learning**: Memory persists across episodes, allowing the

model to accumulate knowledge over time.

3. **Differentiable Memory**: Both reading and writing operations are

differentiable, enabling end-to-end training.

Reference: Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016).
Meta-Learning with Memory-Augmented Neural Networks. ICML.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MANNAlgorithm(MANNOptions<,,>)` | Initializes a new instance of the MANNAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task by writing support examples to memory. |
| `ApplySoftmax(Vector<>)` | Applies softmax to a vector. |
| `CombineWithMemory(Vector<>,Vector<>)` | Combines controller output with memory content. |
| `ComputeAttentionWeights(Vector<>)` | Computes attention weights using cosine similarity. |
| `ComputeLoss(Matrix<>,)` | Computes cross-entropy loss. |
| `ConsolidateMemory` | Applies memory consolidation to prune rarely-used memories. |
| `ExtractFeatureVector(,Int32)` | Extracts feature vector from output at specified index. |
| `GenerateMemoryKey(,Int32)` | Generates a memory key from input at specified index. |
| `GenerateMemoryValue(,Int32)` | Generates a memory value from output label at specified index. |
| `GeneratePrediction(Vector<>)` | Generates final prediction from combined features. |
| `GetBatchSize()` | Gets batch size from input. |
| `GetClassLabel(,Int32)` | Gets class label from output at specified index. |
| `GetOptions` |  |
| `InitializeMemory` | Initializes memory with random or pre-trained values. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using MANN's episodic training with memory. |
| `ProcessQuerySetWithMemory()` | Processes query set using memory-augmented computation. |
| `ProcessWithController(,Int32)` | Processes input through the controller network. |
| `ReadFromMemory(Vector<>)` | Reads relevant content from external memory using attention. |
| `StackPredictions(List<Vector<>>)` | Stacks prediction vectors into a matrix. |
| `TrainEpisode(IMetaLearningTask<,,>)` | Trains the controller on a single episode. |
| `WriteSupportSetToMemory(,,ExternalMemory<>)` | Writes support set examples to external memory. |

