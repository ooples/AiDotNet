# Nested Learning for AiDotNet

Implementation of Google's Nested Learning paradigm for continual learning without catastrophic forgetting.

## Overview

Nested Learning treats ML models as interconnected, multi-level learning problems optimized simultaneously at different timescales. This addresses catastrophic forgetting in continual learning scenarios.

## Key Components

### 1. Continuum Memory System (CMS)

Provides a spectrum of memory modules operating at different frequencies:

- **Level 0 (Fast)**: Updates every step - immediate patterns
- **Level 1 (Medium)**: Updates every 10 steps - tactical patterns
- **Level 2 (Slow)**: Updates every 100 steps - strategic patterns

```csharp
var cms = new ContinuumMemorySystem<double>(
    memoryDimension: 128,
    numFrequencyLevels: 3);

cms.Store(representation, frequencyLevel: 0);
cms.Consolidate(); // Biological memory consolidation
```

### 2. Nested Learner

Main training algorithm with multi-level optimization:

```csharp
var learner = new NestedLearner<double, Matrix<double>, Vector<double>>(
    model: yourModel,
    lossFunction: new MeanSquaredError<double>(),
    numLevels: 3,
    memoryDimension: 128);

// Train on task
var result = learner.Train(trainingData, maxIterations: 1000);

// Adapt to new task without forgetting
var adaptResult = learner.AdaptToNewTask(
    newTaskData,
    preservationStrength: 0.7); // 0.7 = strongly preserve old knowledge
```

### 3. CMS Layer

Neural network layer implementing CMS:

```csharp
var cmsLayer = new ContinuumMemorySystemLayer<double>(
    inputShape: new[] { 256 },
    memoryDim: 256,
    numFrequencyLevels: 3);

var output = cmsLayer.Forward(input);
cmsLayer.ConsolidateMemory();
```

## Example: Continual Learning

```csharp
using AiDotNet.NestedLearning;
using AiDotNet.NeuralNetworks;
using AiDotNet.LinearAlgebra;

// Create model
var model = new FeedForwardNeuralNetwork<double>(architecture);

// Create nested learner
var learner = new NestedLearner<double, Matrix<double>, Vector<double>>(
    model,
    new CrossEntropyLoss<double>(),
    numLevels: 3,
    memoryDimension: 128);

// Train on Task 1
learner.Train(task1Data, maxIterations: 1000);

// Adapt to Task 2 (preserving Task 1)
learner.AdaptToNewTask(task2Data, preservationStrength: 0.7);

// Adapt to Task 3
learner.AdaptToNewTask(task3Data, preservationStrength: 0.7);
```

## Benefits

1. **Prevents Catastrophic Forgetting**: Multi-timescale updates preserve long-term knowledge
2. **Adaptive Learning**: Different levels capture patterns at different timescales
3. **Memory Consolidation**: Mimics biological memory for better retention
4. **Simple Integration**: Works with existing AiDotNet models and infrastructure

## Architecture

### Update Frequencies
- Level 0: Every 1 step (10^0)
- Level 1: Every 10 steps (10^1)
- Level 2: Every 100 steps (10^2)

### Learning Rates
- Level 0: 0.01 (fast adaptation)
- Level 1: 0.001 (medium)
- Level 2: 0.0001 (slow)

### Memory Decay Rates
- Level 0: 0.90 (fast decay)
- Level 1: 0.95 (medium)
- Level 2: 0.99 (slow decay)

## References

- [Google Research Blog: Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- Nested Learning research paper
