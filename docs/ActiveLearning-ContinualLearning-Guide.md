# Active Learning & Continual Learning Guide

## Overview

AiDotNet provides comprehensive implementations of Active Learning (AL) and Continual Learning (CL) strategies. This guide covers how to use these features effectively.

## Active Learning

Active Learning helps you train models more efficiently by strategically selecting which samples to label. Instead of labeling all data, the model identifies the most informative samples.

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **UncertaintySampling** | Selects samples where model is most uncertain | General use, classification |
| **LeastConfidenceSampling** | Selects samples with lowest prediction confidence | Binary/multi-class classification |
| **MarginSampling** | Selects samples with smallest margin between top predictions | Classification with close decisions |
| **EntropySampling** | Selects samples with highest prediction entropy | Multi-class problems |
| **QueryByCommittee** | Uses committee disagreement for selection | Ensemble models |
| **BALD** | Bayesian Active Learning by Disagreement | Probabilistic models |
| **BatchBALD** | BALD extended for batch selection | Batch mode acquisition |
| **DiversitySampling** | Selects diverse samples covering feature space | Avoiding redundancy |
| **CoreSetSelection** | k-Center-Greedy selection in feature space | Large unlabeled pools |
| **DensityWeightedSampling** | Combines uncertainty with density | Avoiding outliers |
| **InformationDensity** | Balances informativeness and representativeness | General use |
| **VariationRatios** | Measures uncertainty via variation ratios | Simple uncertainty measure |
| **ExpectedModelChange** | Selects samples causing largest gradient changes | Deep learning |
| **HybridSampling** | Combines uncertainty and diversity | Production systems |
| **RandomSampling** | Random selection (baseline) | Comparison baseline |

### Basic Usage

```csharp
using AiDotNet.ActiveLearning;
using AiDotNet.Interfaces;

// Create an active learning strategy
var strategy = new UncertaintySampling<double>(
    UncertaintySampling<double>.UncertaintyMeasure.Entropy);

// Use the strategy to select samples from an unlabeled pool
var model = GetYourModel(); // IFullModel<double, Tensor<double>, Tensor<double>>
var unlabeledPool = GetUnlabeledData(); // Tensor<double>
int batchSize = 10;

// Select the most informative samples to label
int[] selectedIndices = strategy.SelectSamples(model, unlabeledPool, batchSize);

// Get selection statistics
var stats = strategy.GetSelectionStatistics();
Console.WriteLine($"Min score: {stats["MinScore"]}");
Console.WriteLine($"Max score: {stats["MaxScore"]}");
Console.WriteLine($"Mean score: {stats["MeanScore"]}");
```

### Strategy-Specific Usage

#### Query by Committee

```csharp
// Create a committee of models
var committee = new List<IFullModel<double, Tensor<double>, Tensor<double>>>
{
    model1,
    model2,
    model3
};

// Create strategy with vote entropy measure
var qbc = new QueryByCommittee<double>(
    committee,
    QueryByCommittee<double>.DisagreementMeasure.VoteEntropy);

int[] selected = qbc.SelectSamples(committee[0], unlabeledPool, batchSize);
```

#### BALD (Bayesian Active Learning)

```csharp
// Create BALD strategy with MC Dropout samples
var bald = new BALD<double>(numMCDropoutSamples: 10);

// Optional: enable batch diversity
bald.UseBatchDiversity = true;

int[] selected = bald.SelectSamples(model, unlabeledPool, batchSize);
```

#### Hybrid Sampling

```csharp
// Combine uncertainty and diversity
var hybrid = new HybridSampling<double>(
    uncertaintyWeight: 0.6,
    diversityWeight: 0.4,
    uncertaintyMeasure: HybridSampling<double>.UncertaintyMeasure.Entropy);

int[] selected = hybrid.SelectSamples(model, unlabeledPool, batchSize);
```

### Active Learning Loop Example

```csharp
public async Task<IFullModel<T, Tensor<T>, Tensor<T>>> ActiveLearningLoop<T>(
    IFullModel<T, Tensor<T>, Tensor<T>> model,
    Tensor<T> unlabeledPool,
    Func<int[], Task<Tensor<T>>> labelingOracle,
    IActiveLearningStrategy<T> strategy,
    int iterations = 10,
    int batchSizePerIteration = 10)
{
    var labeledIndices = new List<int>();
    var labeledInputs = new List<Tensor<T>>();
    var labeledTargets = new List<Tensor<T>>();

    for (int i = 0; i < iterations; i++)
    {
        // Select most informative samples
        int[] selectedIndices = strategy.SelectSamples(
            model, unlabeledPool, batchSizePerIteration);

        // Get labels from oracle (human annotator, etc.)
        Tensor<T> labels = await labelingOracle(selectedIndices);

        // Add to labeled set
        labeledIndices.AddRange(selectedIndices);

        // Retrain model on expanded labeled set
        model.Train(GetInputsForIndices(unlabeledPool, selectedIndices), labels);

        // Log progress
        Console.WriteLine($"Iteration {i + 1}: {labeledIndices.Count} total labeled samples");
    }

    return model;
}
```

## Continual Learning

Continual Learning enables models to learn new tasks without forgetting previously learned knowledge. This is essential for systems that must adapt to new data over time.

### Available Strategies

| Strategy | Description | Memory | Best For |
|----------|-------------|--------|----------|
| **ElasticWeightConsolidation (EWC)** | Fisher Information regularization | Low | Regularization-based |
| **OnlineEWC** | Efficient online version of EWC | Low | Streaming tasks |
| **SynapticIntelligence (SI)** | Online importance estimation | Low | Efficient training |
| **MemoryAwareSynapses (MAS)** | Unsupervised importance estimation | Low | No labels needed |
| **LearningWithoutForgetting (LwF)** | Knowledge distillation | Low | Classification |
| **GradientEpisodicMemory (GEM)** | Gradient projection with memory | Medium | Hard constraints |
| **AveragedGEM (A-GEM)** | Efficient approximation of GEM | Medium | Faster GEM |
| **ExperienceReplay** | Replay buffer of past experiences | High | Simple and effective |
| **GenerativeReplay** | Use generator to replay past data | Medium | Memory efficient |
| **PackNet** | Progressive network pruning | Low | Network compression |
| **ProgressiveNeuralNetworks** | Add new columns for new tasks | High | No forgetting |
| **VariationalContinualLearning (VCL)** | Bayesian posterior updates | Medium | Uncertainty-aware |

### Basic Usage

```csharp
using AiDotNet.ContinualLearning;
using AiDotNet.Interfaces;

// Create a continual learning strategy
var strategy = new ElasticWeightConsolidation<double>(lambda: 5000.0);

// Get your neural network
var network = GetYourNeuralNetwork(); // INeuralNetwork<double>

// Before training on a new task
strategy.BeforeTask(network, taskId: 1);

// Train on the task (your training loop)
TrainOnTask(network, taskData);

// After training on the task
strategy.AfterTask(network, taskData, taskId: 1);

// The strategy now protects important weights learned from task 1
```

### Strategy-Specific Usage

#### Elastic Weight Consolidation

```csharp
// Higher lambda = more protection against forgetting
var ewc = new ElasticWeightConsolidation<double>(
    lambda: 5000.0,
    fisherEstimationSamples: 200);

// Use in training loop
strategy.BeforeTask(network, taskId);
for (int epoch = 0; epoch < epochs; epoch++)
{
    var baseLoss = ComputeTaskLoss(network, taskData);

    // Add EWC regularization loss
    var ewcLoss = ewc.ComputeLoss(network);
    var totalLoss = baseLoss + ewcLoss;

    // Backpropagate and update
    Backpropagate(totalLoss);
}
strategy.AfterTask(network, taskData, taskId);
```

#### Learning without Forgetting

```csharp
var lwf = new LearningWithoutForgetting<double>(
    lambda: 1.0,
    temperature: 2.0);  // Higher temperature = softer probability distribution

// IMPORTANT: LwF requires PrepareDistillation before training on new task
strategy.BeforeTask(network, taskId);
lwf.PrepareDistillation(network, newTaskInputs, taskId);

// Train with distillation loss
for (int epoch = 0; epoch < epochs; epoch++)
{
    var taskLoss = ComputeTaskLoss(network, taskData);
    var distillLoss = lwf.ComputeLoss(network);
    var totalLoss = taskLoss + distillLoss;

    Backpropagate(totalLoss);
}
strategy.AfterTask(network, taskData, taskId);
```

#### Gradient Episodic Memory

```csharp
var gem = new GradientEpisodicMemory<double>(
    memorySize: 256,  // Samples per task
    margin: 0.5);     // Constraint margin

// GEM modifies gradients to prevent forgetting
strategy.BeforeTask(network, taskId);

// In training loop
var gradients = ComputeGradients(network, taskData);

// Project gradients to satisfy constraints from previous tasks
var projectedGradients = gem.ModifyGradients(network, gradients);

ApplyGradients(network, projectedGradients);

strategy.AfterTask(network, taskData, taskId);
```

#### Experience Replay

```csharp
var replay = new ExperienceReplay<double>(
    bufferSize: 1000,
    batchSize: 32);

// After each task, replay stores samples automatically via AfterTask
strategy.AfterTask(network, taskData, taskId);

// During training on new tasks, sample from replay buffer
var replayBatch = replay.SampleBatch(network);
if (replayBatch.HasValue)
{
    var (inputs, targets) = replayBatch.Value;
    // Train on replay batch
    network.Train(inputs, targets);
}
```

### Continual Learning Loop Example

```csharp
public async Task<INeuralNetwork<T>> ContinualLearningLoop<T>(
    INeuralNetwork<T> network,
    List<(Tensor<T> inputs, Tensor<T> targets)> tasks,
    IContinualLearningStrategy<T> strategy,
    int epochsPerTask = 10)
{
    for (int taskId = 0; taskId < tasks.Count; taskId++)
    {
        var (inputs, targets) = tasks[taskId];

        Console.WriteLine($"Learning Task {taskId + 1}/{tasks.Count}");

        // Prepare for new task
        strategy.BeforeTask(network, taskId);

        // Train on task
        for (int epoch = 0; epoch < epochsPerTask; epoch++)
        {
            // Standard task loss
            var output = network.ForwardWithMemory(inputs);
            var taskLoss = ComputeLoss(output, targets);

            // Add continual learning regularization
            var clLoss = strategy.ComputeLoss(network);
            var totalLoss = Add(taskLoss, clLoss);

            // Backprop
            var gradients = network.ComputeGradients(inputs, targets);

            // Optional: modify gradients (for GEM-based methods)
            gradients = strategy.ModifyGradients(network, gradients);

            network.ApplyGradients(gradients, learningRate);
        }

        // Consolidate task knowledge
        strategy.AfterTask(network, (inputs, targets), taskId);

        // Evaluate on all tasks to measure forgetting
        EvaluateOnAllTasks(network, tasks, taskId);
    }

    return network;
}
```

## Combining Active and Continual Learning

Active Learning and Continual Learning can be combined for scenarios where you need to:
1. Efficiently select which samples to label (AL)
2. Learn new tasks without forgetting old ones (CL)

```csharp
public async Task CombinedALCL<T>(
    INeuralNetwork<T> network,
    List<Tensor<T>> taskPools,  // Unlabeled pools for each task
    Func<int, int[], Task<Tensor<T>>> labelingOracle,
    IActiveLearningStrategy<T> alStrategy,
    IContinualLearningStrategy<T> clStrategy)
{
    for (int taskId = 0; taskId < taskPools.Count; taskId++)
    {
        var unlabeledPool = taskPools[taskId];

        // Prepare for continual learning
        clStrategy.BeforeTask(network, taskId);

        // Active Learning: iteratively select and label samples
        var labeledInputs = new List<Tensor<T>>();
        var labeledTargets = new List<Tensor<T>>();

        for (int alIter = 0; alIter < 5; alIter++)
        {
            // Select most informative samples
            int[] selected = alStrategy.SelectSamples(
                network as IFullModel<T, Tensor<T>, Tensor<T>>,
                unlabeledPool,
                batchSize: 10);

            // Get labels
            var labels = await labelingOracle(taskId, selected);

            // Add to labeled set and train
            // ... training code ...
        }

        // Consolidate with continual learning
        var taskData = CombineData(labeledInputs, labeledTargets);
        clStrategy.AfterTask(network, taskData, taskId);
    }
}
```

## Best Practices

### Active Learning

1. **Start with UncertaintySampling** - It's simple, effective, and a good baseline
2. **Use HybridSampling for production** - Combines uncertainty and diversity
3. **Monitor selection statistics** - Track min/max/mean scores to understand model confidence
4. **Consider batch diversity** - Set `UseBatchDiversity = true` to avoid redundant samples
5. **Compare against RandomSampling** - Ensure your strategy provides meaningful improvement

### Continual Learning

1. **Choose based on memory constraints**:
   - Low memory: EWC, SI, MAS
   - Medium memory: GEM, A-GEM, VCL
   - High memory: Experience Replay, Progressive Networks

2. **Tune lambda carefully** - Too high prevents learning, too low causes forgetting

3. **Use Experience Replay as a strong baseline** - Simple and often very effective

4. **Monitor backward transfer** - Track performance on previous tasks over time

5. **Consider task similarity** - Similar tasks need less protection (lower lambda)

## API Reference

### IActiveLearningStrategy<T>

```csharp
public interface IActiveLearningStrategy<T>
{
    string Name { get; }
    bool UseBatchDiversity { get; set; }
    int[] SelectSamples(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> unlabeledPool, int batchSize);
    Dictionary<string, T> GetSelectionStatistics();
}
```

### IContinualLearningStrategy<T>

```csharp
public interface IContinualLearningStrategy<T>
{
    double Lambda { get; set; }
    void BeforeTask(INeuralNetwork<T> network, int taskId);
    void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId);
    T ComputeLoss(INeuralNetwork<T> network);
    Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients);
    void Reset();
}
```

## Further Reading

### Active Learning
- Settles, B. (2012). "Active Learning." Morgan & Claypool.
- Houlsby et al. (2011). "Bayesian Active Learning for Classification and Preference Learning."
- Sener & Savarese (2018). "Active Learning for Convolutional Neural Networks: A Core-Set Approach."

### Continual Learning
- Kirkpatrick et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." (EWC)
- Zenke et al. (2017). "Continual Learning Through Synaptic Intelligence."
- Lopez-Paz & Ranzato (2017). "Gradient Episodic Memory for Continual Learning."
- van de Ven & Tolias (2019). "Three Scenarios for Continual Learning."
