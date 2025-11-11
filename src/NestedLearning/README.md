# Nested Learning for AiDotNet

Implementation of Google's Nested Learning paradigm for continual learning in .NET.

## Overview

Nested Learning is a new machine learning paradigm that treats models as interconnected, multi-level learning problems optimized simultaneously. It addresses the catastrophic forgetting problem in continual learning by operating at multiple timescales with a Continuum Memory System (CMS).

## Key Components

### 1. Continuum Memory System (CMS)

Instead of binary short/long-term memory, CMS provides a **spectrum of memory modules** operating at different frequencies:

- **Fast Memory (Level 0)**: Updates every step, captures immediate patterns
- **Medium Memory (Level 1)**: Updates every 10 steps, captures tactical patterns
- **Slow Memory (Level 2)**: Updates every 100 steps, captures strategic patterns

```csharp
var cms = new ContinuumMemorySystem<double>(
    memoryDimension: 128,
    numFrequencyLevels: 3);

// Store at different frequency levels
cms.Store(representation, frequencyLevel: 0); // Fast memory
cms.Store(representation, frequencyLevel: 2); // Slow memory

// Consolidate memories (mimics biological memory consolidation)
cms.Consolidate();
```

### 2. Context Flow

Maintains distinct information pathways and update rates for each nested optimization level:

```csharp
var contextFlow = new ContextFlow<double>(
    contextDimension: 128,
    numLevels: 3);

// Propagate context through levels
var context = contextFlow.PropagateContext(input, currentLevel: 0);

// Compute gradients for multi-level optimization
var gradients = contextFlow.ComputeContextGradients(upstreamGradient, level: 1);
```

### 3. Nested Learner

The main training algorithm that coordinates multi-level optimization:

```csharp
var nestedLearner = new NestedLearner<double, Matrix<double>, Vector<double>>(
    model: yourModel,
    lossFunction: new MeanSquaredError<double>(),
    numLevels: 3,
    memoryDimension: 128);

// Train with nested learning
var result = nestedLearner.Train(
    trainingData,
    numLevels: 3,
    maxIterations: 1000);

// Adapt to new task without forgetting
var adaptResult = nestedLearner.AdaptToNewTask(
    newTaskData,
    preservationStrength: 0.5); // 0 = forget everything, 1 = preserve everything
```

### 4. Hope Architecture

Self-modifying recurrent neural network based on the Titans architecture with CMS blocks:

```csharp
var architecture = new NeuralNetworkArchitecture<double>();
var hope = new HopeNetwork<double>(
    architecture,
    hiddenDim: 256,
    numCMSLevels: 3,
    numRecurrentLayers: 2);

// Add output layer
hope.AddOutputLayer(outputDim: 10, ActivationFunction.Softmax);

// Forward pass with self-referential optimization
var output = hope.Forward(input);

// Periodically consolidate memories
hope.ConsolidateMemory();
```

### 5. CMS Layer

Neural network layer implementing Continuum Memory System:

```csharp
var cmsLayer = new ContinuumMemorySystemLayer<double>(
    inputShape: new[] { 256 },
    memoryDim: 256,
    numFrequencyLevels: 3);

// Use in neural network
var output = cmsLayer.Forward(input);

// Access memory states
var memoryStates = cmsLayer.GetMemoryStates();

// Reset memory
cmsLayer.ResetMemory();
```

## Example: Continual Learning

```csharp
using AiDotNet.NestedLearning;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;

// Setup model
var architecture = new NeuralNetworkArchitecture<double>();
var model = new FeedForwardNeuralNetwork<double>(architecture);

// Create nested learner
var learner = new NestedLearner<double, Matrix<double>, Vector<double>>(
    model: model,
    lossFunction: new CrossEntropyLoss<double>(),
    numLevels: 3,
    memoryDimension: 128);

// Train on Task 1
var task1Result = learner.Train(task1Data, maxIterations: 1000);
Console.WriteLine($"Task 1 Loss: {task1Result.FinalLoss}");

// Adapt to Task 2 (preserving Task 1 knowledge)
var task2Result = learner.AdaptToNewTask(
    task2Data,
    preservationStrength: 0.7); // Strongly preserve Task 1

Console.WriteLine($"Task 2 Loss: {task2Result.NewTaskLoss}");
Console.WriteLine($"Forgetting Metric: {task2Result.ForgettingMetric}");

// Adapt to Task 3
var task3Result = learner.AdaptToNewTask(task3Data, preservationStrength: 0.7);
```

## Example: Hope Network for Sequence Modeling

```csharp
using AiDotNet.NestedLearning;
using AiDotNet.NeuralNetworks;

// Create Hope architecture
var architecture = new NeuralNetworkArchitecture<double>();
var hope = new HopeNetwork<double>(
    architecture,
    hiddenDim: 512,
    numCMSLevels: 4,      // More levels for longer-term patterns
    numRecurrentLayers: 3); // Deep recurrent processing

hope.AddOutputLayer(outputDim: vocabSize, ActivationFunction.Softmax);

// Training loop
foreach (var sequence in sequences)
{
    hope.ResetRecurrentState(); // Reset for new sequence

    foreach (var (input, target) in sequence)
    {
        var output = hope.Forward(input);
        var loss = lossFunction.ComputeLoss(output, target);
        var gradient = lossFunction.ComputeGradient(output, target);
        hope.Backward(gradient);
    }

    // Consolidate memories every 100 sequences
    if (sequenceCount % 100 == 0)
    {
        hope.ConsolidateMemory();
    }
}
```

## Key Benefits

1. **Prevents Catastrophic Forgetting**: Multi-timescale updates preserve long-term knowledge while adapting to new tasks

2. **Adaptive Learning**: Different optimization levels capture patterns at different timescales

3. **Memory Consolidation**: Mimics biological memory consolidation for better continual learning

4. **Self-Referential Optimization**: Hope architecture adapts its processing based on meta-state

5. **Scalable Context Windows**: CMS enables extended context windows for sequence modeling

## Research References

- **Google Research Blog**: [Introducing Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)
- **Paper**: Nested Learning - A New ML Paradigm for Continual Learning

## Architecture Details

### Update Frequencies

By default, nested learning uses exponentially increasing update frequencies:

- Level 0: Every 1 step (10^0)
- Level 1: Every 10 steps (10^1)
- Level 2: Every 100 steps (10^2)
- Level 3: Every 1000 steps (10^3)

### Learning Rates

Learning rates decrease exponentially with level:

- Level 0: 0.01 (fast adaptation)
- Level 1: 0.001 (medium adaptation)
- Level 2: 0.0001 (slow adaptation)

### Memory Decay Rates

Decay rates increase with level (slower levels decay more slowly):

- Level 0: 0.5 (fast decay)
- Level 1: 0.25 (medium decay)
- Level 2: 0.125 (slow decay)

## Integration with AiDotNet

Nested Learning integrates seamlessly with AiDotNet's existing infrastructure:

- Implements `INestedLearner<T, TInput, TOutput>` interface
- Works with any `IFullModel<T, TInput, TOutput>`
- Compatible with all AiDotNet loss functions
- Uses standard `Tensor<T>` and `Vector<T>` types
- Follows AiDotNet's architecture patterns

## License

Part of the AiDotNet library. See main repository for license information.
