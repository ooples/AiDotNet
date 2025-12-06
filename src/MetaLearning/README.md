# AiDotNet Meta-Learning Framework

This module implements production-ready meta-learning algorithms for few-shot learning in .NET.

## Overview

Meta-learning (learning to learn) enables models to quickly adapt to new tasks with minimal examples. This implementation includes:

- **SEAL (Sample-Efficient Adaptive Learning)**: Enhanced meta-learning with temperature scaling and adaptive learning rates
- **MAML (Model-Agnostic Meta-Learning)**: The foundational gradient-based meta-learning algorithm
- **Reptile**: A simpler, more efficient alternative to MAML
- **iMAML (implicit MAML)**: Memory-efficient variant using implicit differentiation

## Key Features

✅ **N-way K-shot Support**: Flexible episodic data interfaces
✅ **Configurable Hyperparameters**: All algorithms support extensive customization
✅ **Checkpointing**: Save and resume training with full state management
✅ **Deterministic Seeding**: Reproducible experiments
✅ **MetaTrainer**: High-level training orchestration with early stopping
✅ **Comprehensive Tests**: ≥90% test coverage with E2E smoke tests

## Quick Start

### Basic Example: 5-way 1-shot Classification

```csharp
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Training;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

// 1. Create a base model (neural network)
var architecture = new NeuralNetworkArchitecture<double>
{
    InputSize = 784,  // e.g., 28x28 images
    OutputSize = 5,   // 5-way classification
    HiddenLayerSizes = new[] { 128, 64 },
    ActivationFunctionType = ActivationFunctionType.ReLU,
    OutputActivationFunctionType = ActivationFunctionType.Softmax,
    TaskType = TaskType.Classification
};

var baseModel = new NeuralNetworkModel<double>(
    new NeuralNetwork<double>(architecture));

// 2. Configure SEAL algorithm
var sealOptions = new SEALAlgorithmOptions<double, Matrix<double>, Vector<double>>
{
    BaseModel = baseModel,
    InnerLearningRate = 0.01,      // Adaptation learning rate
    OuterLearningRate = 0.001,     // Meta-learning rate
    AdaptationSteps = 5,            // Gradient steps per task
    MetaBatchSize = 4,              // Tasks per meta-update
    Temperature = 1.0,
    UseFirstOrder = true,           // Efficient gradient computation
    RandomSeed = 42
};

var algorithm = new SEALAlgorithm<double, Matrix<double>, Vector<double>>(sealOptions);

// 3. Create episodic datasets (implement IEpisodicDataset)
var trainDataset = new MyEpisodicDataset(split: DatasetSplit.Train);
var valDataset = new MyEpisodicDataset(split: DatasetSplit.Validation);

// 4. Configure trainer
var trainerOptions = new MetaTrainerOptions
{
    NumEpochs = 100,
    TasksPerEpoch = 1000,
    MetaBatchSize = 4,
    NumWays = 5,             // 5 classes per task
    NumShots = 1,            // 1 example per class
    NumQueryPerClass = 15,   // 15 query examples per class
    CheckpointInterval = 10,
    CheckpointDir = "./checkpoints",
    EarlyStoppingPatience = 20,
    RandomSeed = 42
};

var trainer = new MetaTrainer<double, Matrix<double>, Vector<double>>(
    algorithm, trainDataset, valDataset, trainerOptions);

// 5. Train
var history = trainer.Train();

// 6. Adapt to a new task
var newTask = testDataset.SampleTasks(1, 5, 1, 15)[0];
var adaptedModel = algorithm.Adapt(newTask);
var predictions = adaptedModel.Predict(newTask.QueryInput);
```

## Algorithm Comparison

| Algorithm | Memory | Speed | Performance | Use Case |
|-----------|--------|-------|-------------|----------|
| **SEAL** | Medium | Medium | High | Best overall performance |
| **MAML** | High | Slow | High | Strong theoretical foundation |
| **Reptile** | Low | Fast | Good | Large-scale applications |
| **iMAML** | Low | Medium | High | Deep adaptation required |

## Episodic Dataset Interface

Implement `IEpisodicDataset<T, TInput, TOutput>` to create custom datasets:

```csharp
public class OmniglotDataset : IEpisodicDataset<double, Matrix<double>, Vector<double>>
{
    public ITask<double, Matrix<double>, Vector<double>>[] SampleTasks(
        int numTasks, int numWays, int numShots, int numQueryPerClass)
    {
        // Your implementation:
        // 1. Randomly select numWays classes
        // 2. Sample numShots examples per class for support set
        // 3. Sample numQueryPerClass examples per class for query set
        // 4. Return array of Task objects
    }

    // ... other interface members
}
```

## Configuration Options

### Inner vs Outer Loop

- **Inner Loop**: Fast adaptation to a specific task
  - Controlled by `InnerLearningRate` and `AdaptationSteps`
  - Uses support set for few-shot learning

- **Outer Loop**: Meta-learning across tasks
  - Controlled by `OuterLearningRate` and `MetaBatchSize`
  - Uses query set for meta-gradient computation

### Hyperparameter Tuning

**For better adaptation**:
- Increase `AdaptationSteps` (5-10)
- Tune `InnerLearningRate` (0.001-0.1)

**For faster meta-learning**:
- Increase `MetaBatchSize` (4-32)
- Increase `OuterLearningRate` (0.0001-0.01)

**For stability**:
- Enable gradient clipping (SEAL)
- Use first-order approximation
- Add weight decay regularization

## Testing

The framework includes comprehensive unit tests:

```bash
dotnet test --filter "FullyQualifiedName~MetaLearning"
```

### Test Coverage

- ✅ Task and TaskBatch creation and validation
- ✅ Episodic dataset sampling
- ✅ SEAL algorithm training and adaptation
- ✅ MAML algorithm training and adaptation
- ✅ Reptile algorithm training and adaptation
- ✅ iMAML algorithm training and adaptation
- ✅ MetaTrainer with checkpointing
- ✅ E2E 5-way 1-shot smoke tests

## Architecture

```
src/MetaLearning/
├── Algorithms/
│   ├── IMetaLearningAlgorithm.cs    # Core interface
│   ├── MetaLearningBase.cs          # Shared base class
│   ├── SEALAlgorithm.cs             # SEAL implementation
│   ├── MAMLAlgorithm.cs             # MAML implementation
│   ├── ReptileAlgorithm.cs          # Reptile implementation
│   └── iMAMLAlgorithm.cs            # iMAML implementation
├── Data/
│   ├── ITask.cs                      # Task interface
│   ├── Task.cs                       # Task implementation
│   ├── IEpisodicDataset.cs          # Dataset interface
│   └── TaskBatch.cs                  # Task batching
└── Training/
    └── MetaTrainer.cs                # Training orchestration

src/Models/Options/
├── MetaLearningAlgorithmOptions.cs   # Base options
├── SEALAlgorithmOptions.cs           # SEAL-specific
├── MAMLAlgorithmOptions.cs           # MAML-specific
├── ReptileAlgorithmOptions.cs        # Reptile-specific
└── iMAMLAlgorithmOptions.cs          # iMAML-specific
```

## References

1. **MAML**: Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. ICML.

2. **Reptile**: Nichol, A., Achiam, J., & Schulman, J. (2018). On first-order meta-learning algorithms. arXiv:1803.02999.

3. **iMAML**: Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019). Meta-learning with implicit gradients. NeurIPS.

## Contributing

This implementation follows AiDotNet conventions:
- Generic type parameters `<T>` for numeric operations
- Comprehensive XML documentation
- Beginner-friendly explanations
- ≥90% test coverage requirement

## License

Apache 2.0 - Same as AiDotNet parent project
