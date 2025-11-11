# Nested Learning for AiDotNet

Production-ready implementation of Google's Nested Learning paradigm - a new approach to continual learning that prevents catastrophic forgetting.

## Overview

Nested Learning treats ML models as **interconnected, multi-level learning problems optimized simultaneously**. This paradigm shift enables:

- **Continual learning without catastrophic forgetting**
- **Multi-timescale optimization** (fast, medium, slow updates)
- **Self-referential optimization** (models optimize their own memory)
- **Deeper computational depth** through context compression

## Core Components

### 1. Hope Architecture

Self-modifying recurrent variant of Titans with **unbounded levels of in-context learning**:

```csharp
var architecture = new NeuralNetworkArchitecture<double>();
var hope = new HopeNetwork<double>(
    architecture,
    hiddenDim: 512,
    numCMSLevels: 4,              // CMS blocks for context scaling
    numRecurrentLayers: 3,        // Recurrent processing depth
    inContextLearningLevels: 5);  // Unbounded in theory, 5 in practice

hope.AddOutputLayer(outputDim: vocabSize, ActivationFunction.Softmax);

// Forward pass with self-referential optimization
var output = hope.Forward(input);

// Model optimizes its own memory through looped learning
hope.ConsolidateMemory();
```

**Key Features:**
- **Self-modification**: Model adjusts its own parameters during inference
- **Looped learning levels**: Infinite, recursive optimization structure
- **CMS blocks**: Extended context windows via Continuum Memory System
- **Associative memory**: Backpropagation modeled as memory (data → local error)

### 2. Continuum Memory System (CMS)

Spectrum of memory modules operating at different frequencies (not binary short/long-term).

**Two implementations available:**

**A. ContinuumMemorySystem<T> - Utility Class (NOT from paper)**
Used by `NestedLearner` for meta-learning. Uses exponential moving averages with decay rates:

```csharp
var cms = new ContinuumMemorySystem<double>(
    memoryDimension: 128,
    numFrequencyLevels: 4);

// Store at different frequencies
cms.Store(representation, frequencyLevel: 0); // Fast (updates every step)
cms.Store(representation, frequencyLevel: 2); // Slow (updates every 100 steps)

// Biological memory consolidation
cms.Consolidate(); // Transfers information from fast → slow memories
```

**B. ContinuumMemorySystemLayer<T> - Paper-Accurate Implementation (Equations 30-31)**
Used by `HopeNetwork` for the HOPE architecture. Uses gradient accumulation with Modified GD:

```csharp
// Used internally by HopeNetwork - implements Equation 31 from paper
var hope = new HopeNetwork<double>(
    architecture,
    hiddenDim: 512,
    numCMSLevels: 4);  // Uses ContinuumMemorySystemLayer internally
```

**Update Frequencies (both implementations):**
- Level 0 (Fast): Every 1 step - immediate patterns
- Level 1 (Medium): Every 10 steps - tactical patterns
- Level 2 (Slow): Every 100 steps - strategic patterns
- Level 3+ (Very Slow): Every 1000+ steps - long-term knowledge

**Memory Update Methods:**
- `ContinuumMemorySystem<T>`: Exponential moving averages with decay rates (utility class)
- `ContinuumMemorySystemLayer<T>`: Gradient accumulation + Modified GD (paper-accurate HOPE)

### 3. Context Flow

Distinct information pathways for each optimization level, enabling deeper computational depth:

```csharp
var contextFlow = new ContextFlow<double>(
    contextDimension: 128,
    numLevels: 5);

// Propagate context through distinct pathways
var flowedContext = contextFlow.PropagateContext(input, currentLevel: 2);

// Compress internal context flows (key innovation)
var compressed = contextFlow.CompressContext(flowedContext, targetLevel: 2);

// Compute gradients through context flow
var gradients = contextFlow.ComputeContextGradients(upstreamGrad, level: 2);
```

**Key Insight:** Deep learning works by "compressing internal context flows". Context Flow makes this explicit, allowing models to be designed with **deeper computational depth**.

### 4. Associative Memory Framework

Models both backpropagation and attention as associative memory:

```csharp
var memory = new AssociativeMemory<double>(
    dimension: 128,
    capacity: 10000);

// Backpropagation as associative memory:
// Maps data point → local error
memory.Associate(dataPoint, localError);

// Attention as associative memory:
// Maps query → key-value pairs
var retrieved = memory.Retrieve(query);
```

**Unifies:**
- Backpropagation (training process as memory)
- Transformer attention (architectural component as memory)
- Enables new optimizer designs with better data sample relationships

### 5. Nested Learner

Main training algorithm with multi-level optimization:

```csharp
var learner = new NestedLearner<double, Matrix<double>, Vector<double>>(
    model: yourModel,
    lossFunction: new CrossEntropyLoss<double>(),
    numLevels: 4,
    memoryDimension: 128);

// Train on Task 1
var result = learner.Train(task1Data, maxIterations: 1000);
Console.WriteLine($"Converged: {result.Converged}, Loss: {result.FinalMetaLoss}");

// Adapt to Task 2 without forgetting Task 1
var adaptResult = learner.AdaptToNewTask(
    task2Data,
    preservationStrength: 0.7); // 0.7 = strongly preserve old knowledge

Console.WriteLine($"Forgetting metric: {adaptResult.ForgettingMetric}");
```

## Complete Example: Continual Learning

```csharp
using AiDotNet.NestedLearning;
using AiDotNet.NeuralNetworks;
using AiDotNet.LinearAlgebra;

// Create feedforward network
var architecture = new NeuralNetworkArchitecture<double>
{
    InputSize = 784,
    OutputSize = 10,
    HiddenLayerSizes = new[] { 256, 128 }
};
var model = new FeedForwardNeuralNetwork<double>(architecture);

// Create nested learner (multi-level optimization)
var learner = new NestedLearner<double, Matrix<double>, Vector<double>>(
    model,
    new CrossEntropyLoss<double>(),
    numLevels: 4,              // 4 nested optimization levels
    memoryDimension: 256);      // Rich memory capacity

// Learn Task 1: Digits 0-4
learner.Train(digits_0_4_data, maxIterations: 1000);

// Learn Task 2: Digits 5-9 (without forgetting 0-4)
learner.AdaptToNewTask(
    digits_5_9_data,
    preservationStrength: 0.8); // Strong preservation

// Learn Task 3: Fashion items
learner.AdaptToNewTask(
    fashion_data,
    preservationStrength: 0.7);

// Model now knows all three tasks without catastrophic forgetting!
```

## Example: Hope Architecture for Sequence Modeling

```csharp
var architecture = new NeuralNetworkArchitecture<double>
{
    InputSize = embedDim,
    OutputSize = vocabSize
};

var hope = new HopeNetwork<double>(
    architecture,
    hiddenDim: 1024,
    numCMSLevels: 6,              // More levels for longer contexts
    numRecurrentLayers: 4,        // Deep recurrent processing
    inContextLearningLevels: 8);  // Unbounded in-context learning

hope.AddOutputLayer(vocabSize, ActivationFunction.Softmax);
hope.SetSelfModificationRate(0.005); // Control self-referential optimization

// Training loop with self-referential optimization
foreach (var sequence in sequences)
{
    hope.ResetRecurrentState(); // New sequence

    foreach (var (input, target) in sequence)
    {
        var output = hope.Forward(input);
        var loss = lossFunction.ComputeLoss(output, target);
        var grad = lossFunction.ComputeGradient(output, target);
        hope.Backward(grad);

        // Model optimizes its own memory through self-referential process
    }

    // Consolidate memories every N sequences
    if (seqCount % 100 == 0)
    {
        hope.ConsolidateMemory();
    }
}

// Access internal mechanisms
var metaState = hope.GetMetaState();
var contextFlow = hope.GetContextFlow();
var associativeMemory = hope.GetAssociativeMemory();
```

## Key Benefits

### 1. **Prevents Catastrophic Forgetting**
Multi-timescale updates preserve long-term knowledge while adapting to new tasks:
- Fast levels (0-1): Learn new task specifics quickly
- Medium levels (2-3): Balance old and new knowledge
- Slow levels (4+): Protect core competencies

### 2. **Self-Referential Optimization**
Hope architecture can optimize its own memory through looped learning:
- Model adjusts its internal representations during inference
- Unbounded levels of in-context learning
- More adaptive than traditional architectures

### 3. **Deeper Computational Depth**
Context compression enables learning components with deeper processing:
- Traditional: Single information pathway
- Nested: Multiple distinct pathways per level
- Result: Richer representations, better performance

### 4. **Unified Framework**
Treats architecture and optimization as single coherent system:
- Backpropagation → Associative memory
- Attention → Associative memory
- Optimization → Nested levels with distinct frequencies

## Architecture Details

### Update Frequencies (by level)
```
Level 0: 10^0 = 1 step    (fastest)
Level 1: 10^1 = 10 steps
Level 2: 10^2 = 100 steps
Level 3: 10^3 = 1000 steps
Level 4+: 10^n steps      (slowest)
```

### Learning Rates (by level)
```
Level 0: 0.01      (fast adaptation)
Level 1: 0.001     (medium)
Level 2: 0.0001    (slow)
Level 3: 0.00001   (very slow)
```

### Memory Decay Rates (ContinuumMemorySystem only)

**IMPORTANT**: These decay rates apply ONLY to the `ContinuumMemorySystem<T>` utility class used by `NestedLearner`. They are **NOT from the research paper** and **NOT used in the HOPE architecture**.

The paper-accurate **HOPE architecture** (via `HopeNetwork<T>`) uses `ContinuumMemorySystemLayer<T>`, which implements **gradient accumulation** (Equation 31) with **Modified Gradient Descent** (Equations 27-29) as specified in the research paper, **not exponential moving averages**.

Formula for `ContinuumMemorySystem<T>`: `updated = (currentMemory × decay) + (newRepresentation × (1 - decay))`

```
Level 0: 0.90  (90% retention, 10% decay per update - moderate persistence)
Level 1: 0.95  (95% retention, 5% decay per update  - high persistence)
Level 2: 0.99  (99% retention, 1% decay per update  - very high persistence)
Level 3: 0.995 (99.5% retention, 0.5% decay per update - extremely high persistence)
```

**Interpretation**: The decay parameter is a *retention factor*. Higher values = more retention of old memory = slower decay rate. Level 3 (0.995) changes very slowly and maintains long-term information, while Level 0 (0.90) adapts more quickly to new inputs.

**For HOPE architecture**: Use `HopeNetwork<T>` which internally uses `ContinuumMemorySystemLayer<T>` with gradient accumulation (Equation 31 from paper), not these decay rates.

## Performance Benchmarks

Based on Google's research, Hope demonstrates:
- **Lower perplexity** vs Titans, Samba, Transformers on language modeling
- **Superior long-context performance** on Needle-In-Haystack tests
- **More efficient context handling** with CMS blocks
- **Better continual learning** with reduced catastrophic forgetting

## Integration with AiDotNet

Nested Learning follows AiDotNet patterns throughout:
- Uses `Vector<T>`, `Matrix<T>`, `Tensor<T>` from `AiDotNet.LinearAlgebra`
- Implements `INumericOperations<T>` pattern with `_numOps`
- Integrates with `IFullModel<T, TInput, TOutput>`
- Returns `MetaTrainingResult<T>`, `MetaAdaptationResult<T>`
- Compatible with all AiDotNet loss functions and optimizers

## References

- [Google Research Blog: Introducing Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- Nested Learning: A New ML Paradigm for Continual Learning (Research Paper)
- Titans Architecture (Foundation for Hope)

## License

Part of the AiDotNet library. See main repository for license information.
