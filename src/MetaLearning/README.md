# AiDotNet Meta-Learning Framework

This directory contains the implementation of meta-learning algorithms for few-shot learning, including MAML, Reptile, and SEAL (Self-Adapting Meta-Learning).

## Overview

Meta-learning enables models to "learn how to learn" by training on many tasks, allowing rapid adaptation to new tasks with minimal examples. This framework supports:

- **N-way K-shot learning**: Train models that can classify N classes with only K examples per class
- **Episodic training**: Task-based training with support and query sets
- **Multiple algorithms**: MAML, Reptile, and SEAL-inspired self-adapting meta-learner
- **Standard benchmarks**: Omniglot and MiniImageNet datasets

## Project Structure

```
MetaLearning/
├── Algorithms/         # Meta-learning algorithm implementations
│   └── IMetaLearner.cs
├── Config/            # Configuration classes for algorithms
│   ├── IMetaLearnerConfig.cs
│   ├── MetaLearnerConfig.cs      # Base config
│   ├── MAMLConfig<T>             # MAML-specific config
│   ├── ReptileConfig<T>          # Reptile-specific config
│   └── SEALConfig<T>             # SEAL-specific config
├── Datasets/          # Dataset interfaces and loaders
│   ├── IMetaDataset.cs
│   └── MetaDatasetBase.cs
├── Samplers/          # Task and episode samplers
│   ├── ITaskSampler.cs
│   └── EpisodicTaskSampler.cs
├── Tasks/             # Episodic task representations
│   ├── IEpisode.cs
│   └── Episode.cs
├── Training/          # Training infrastructure (future)
└── Evaluation/        # Evaluation harness (future)
```

## Core Concepts

### Episodes (N-way K-shot)

An episode is a mini-training task consisting of:
- **Support set**: K examples per class for adaptation (training)
- **Query set**: Q examples per class for evaluation (testing)
- **N ways**: Number of classes in the episode
- **K shots**: Number of support examples per class

Example: 5-way 1-shot learning
- 5 classes randomly sampled
- 1 support example per class (5 total)
- 15 query examples per class (75 total)

### Meta-Learning Loop

Meta-learning uses a two-loop structure:

1. **Inner Loop** (Task Adaptation):
   - Fast adaptation to a specific task using the support set
   - Few gradient steps with higher learning rate
   - Goal: Solve the current task

2. **Outer Loop** (Meta-Optimization):
   - Update meta-parameters based on query set performance
   - Improve the adaptation process across tasks
   - Goal: Learn how to adapt quickly

## Implemented Components

### ✅ Milestone 1 - Core Design (Complete)

- **IEpisode<T>**: Interface for N-way K-shot episodes
- **Episode<T>**: Default episode implementation with validation
- **IMetaDataset<T>**: Interface for episodic dataset access
- **MetaDatasetBase<T>**: Base class with sampling utilities
- **ITaskSampler<T>**: Interface for sampling episodes
- **EpisodicTaskSampler<T>**: Implementation with Fisher-Yates sampling
- **IMetaLearner<T>**: Core meta-learner interface
- **Configuration classes**: For MAML, Reptile, and SEAL

### ✅ Milestone 2 - Data Abstractions (Partial)

- **Base dataset infrastructure**: MetaDatasetBase<T>
- **Task sampler**: EpisodicTaskSampler<T> with reproducible sampling

### 🔄 Remaining Work

#### Milestone 2 - Complete Data Infrastructure
- [ ] Omniglot dataset loader
- [ ] MiniImageNet dataset loader
- [ ] Synthetic dataset for testing
- [ ] Unit tests for data abstractions

#### Milestone 3 - Algorithm Implementations
- [ ] MAML (Model-Agnostic Meta-Learning)
- [ ] Reptile (First-order meta-learning)
- [ ] SEAL (Self-Adapting Meta-Learning)
- [ ] MetaTrainer with checkpointing and logging
- [ ] Unit tests for algorithms

#### Milestone 4 - Evaluation & Benchmarks
- [ ] Evaluation harness with train/val/test splits
- [ ] Benchmarks on Omniglot (5-way 1-shot, 5-way 5-shot)
- [ ] Benchmarks on MiniImageNet
- [ ] Comparison metrics and reports

#### Milestone 5 - Documentation
- [ ] User guide: "Train SEAL on Omniglot/MiniImageNet"
- [ ] Example scripts and configurations
- [ ] API documentation

#### Milestone 6 - Testing & Integration
- [ ] E2E smoke test: 5-way 1-shot Omniglot
- [ ] CI integration
- [ ] Regression checks

## Algorithm Details

### MAML (Model-Agnostic Meta-Learning)

**Key Features:**
- Learns an initialization for rapid adaptation
- Uses second-order gradients (or first-order approximation)
- Meta-updates based on query set performance

**Configuration:**
```csharp
var config = new MAMLConfig<double>
{
    InnerLearningRate = 0.01,      // Task adaptation LR
    OuterLearningRate = 0.001,     // Meta-optimization LR
    InnerSteps = 5,                 // Adaptation steps
    MetaBatchSize = 4,              // Tasks per meta-update
    FirstOrder = false              // Use second-order gradients
};
```

### Reptile

**Key Features:**
- Simpler alternative to MAML
- First-order only (no second-order gradients)
- Moves meta-parameters toward adapted parameters
- Often performs similarly to MAML

**Configuration:**
```csharp
var config = new ReptileConfig<double>
{
    InnerLearningRate = 0.01,
    OuterLearningRate = 0.001,
    InnerSteps = 10,                // More steps than MAML
    MetaBatchSize = 1,              // Often uses single tasks
    Epsilon = 1.0                   // Interpolation coefficient
};
```

### SEAL (Self-Adapting Meta-Learning)

**Key Features:**
- Adaptive inner learning rates based on task difficulty
- Self-improvement mechanism for refining adaptation strategy
- Task embeddings for conditioning adaptation
- Temperature-based adaptation weighting
- Regularization to prevent overfitting during adaptation

**Configuration:**
```csharp
var config = new SEALConfig<double>
{
    InnerLearningRate = 0.01,
    OuterLearningRate = 0.001,
    InnerSteps = 5,
    MetaBatchSize = 4,
    Temperature = 1.0,
    AdaptiveInnerLR = true,         // Adaptive learning rates
    UseSelfImprovement = true,      // Self-improvement mechanism
    SelfImprovementSteps = 3,
    AdaptationRegularization = 0.01,
    UseTaskEmbeddings = true,       // Task conditioning
    TaskEmbeddingDim = 64
};
```

## Usage Examples

### Creating an Episode

```csharp
var episode = new Episode<double>(
    supportData: supportTensor,      // [N*K, ...]
    supportLabels: supportLabels,    // [N*K]
    queryData: queryTensor,          // [N*Q, ...]
    queryLabels: queryLabels,        // [N*Q]
    numWays: 5,
    numShots: 1,
    numQueries: 15
);
```

### Sampling Episodes

```csharp
var sampler = new EpisodicTaskSampler<double>(
    dataset: omniglotDataset,
    numWays: 5,
    numShots: 1,
    numQueries: 15,
    seed: 42
);

// Sample single episode
var episode = sampler.SampleEpisode();

// Sample batch of episodes
var episodes = sampler.SampleBatch(batchSize: 4);
```

### Meta-Learning Training Loop (Conceptual)

```csharp
// Create meta-learner
var metaLearner = new MAML<double>(baseModel, config);

// Training loop
for (int iteration = 0; iteration < maxIterations; iteration++)
{
    // Sample batch of tasks
    var episodes = trainSampler.SampleBatch(config.MetaBatchSize);

    // Meta-training step
    var metrics = metaLearner.MetaTrainStep(episodes);

    Console.WriteLine($"Iteration {iteration}: Loss={metrics.MetaLoss}, Accuracy={metrics.Accuracy}");

    // Periodic evaluation
    if (iteration % 100 == 0)
    {
        var valEpisodes = valSampler.SampleBatch(100);
        var evalMetrics = metaLearner.Evaluate(valEpisodes);
        Console.WriteLine($"Validation Accuracy: {evalMetrics.Accuracy} ± {evalMetrics.AccuracyStd}");
    }
}
```

## Metrics and Evaluation

### Training Metrics

- **MetaLoss**: Outer loop loss for meta-optimization
- **TaskLoss**: Average inner loop loss across episodes
- **Accuracy**: Query set accuracy after adaptation

### Evaluation Metrics

- **Accuracy**: Mean accuracy across episodes
- **AccuracyStd**: Standard deviation of accuracy
- **ConfidenceInterval**: 95% confidence interval
- **Loss**: Mean query set loss

### Adaptation Metrics

- **QueryAccuracy**: Query performance after adaptation
- **SupportAccuracy**: Support performance (should be high)
- **AdaptationSteps**: Number of steps taken
- **AdaptationTimeMs**: Time taken for adaptation

## Datasets

### Omniglot

- **Description**: 1,623 handwritten characters from 50 alphabets
- **Examples per class**: 20
- **Standard splits**:
  - Train: 1,200 classes
  - Validation: 100 classes
  - Test: 423 classes
- **Common benchmarks**: 5-way 1-shot, 5-way 5-shot, 20-way 1-shot

### MiniImageNet

- **Description**: 100 classes from ImageNet
- **Examples per class**: 600
- **Image size**: 84×84×3
- **Standard splits**:
  - Train: 64 classes
  - Validation: 16 classes
  - Test: 20 classes
- **Common benchmarks**: 5-way 1-shot, 5-way 5-shot

## Integration with AiDotNet

This framework integrates seamlessly with existing AiDotNet components:

- **Tensors**: Uses `AiDotNet.LinearAlgebra.Tensor<T>`
- **Layers**: Compatible with `ILayer<T>` interface
- **Models**: Works with `INeuralNetwork<T>`
- **Operations**: Uses `INumericOperations<T>` for type-agnostic math

## References

### Papers

1. **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
2. **Reptile**: Nichol et al., "On First-Order Meta-Learning Algorithms", arXiv 2018
3. **SEAL**: MIT CSAIL, "Self-Adapting Language Models", arXiv 2506.10943

### Resources

- MAML TensorFlow: https://github.com/cbfinn/maml
- Reptile OpenAI: https://github.com/openai/supervised-reptile
- SEAL MIT: https://github.com/Continual-Intelligence/SEAL

## Development Status

| Milestone | Status | Completion |
|-----------|--------|------------|
| M1: Core Design & APIs | ✅ Complete | 100% |
| M2: Data Abstractions | 🔄 In Progress | 40% |
| M3: Algorithms | ⏳ Pending | 0% |
| M4: Evaluation | ⏳ Pending | 0% |
| M5: Documentation | ⏳ Pending | 0% |
| M6: Testing & CI | ⏳ Pending | 0% |

## Next Steps

1. **Complete M2**: Implement Omniglot and MiniImageNet loaders
2. **Start M3**: Implement MAML as the first baseline
3. **Implement Reptile**: Second baseline for comparison
4. **Implement SEAL**: Advanced self-adapting meta-learner
5. **Build MetaTrainer**: Training infrastructure with logging
6. **Add Tests**: Unit and integration tests
7. **Run Benchmarks**: Evaluate on standard few-shot tasks
8. **Documentation**: Complete user guide and examples

## Contributing

When extending this framework:

1. Follow existing patterns (IMetaLearner, IMetaDataset)
2. Add comprehensive XML documentation
3. Include "For Beginners" sections
4. Write unit tests for new components
5. Update this README with new features

## License

This implementation is part of AiDotNet and follows the project's license.
Ensure compatibility with referenced algorithms and datasets.
