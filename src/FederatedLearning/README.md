# Federated Learning Framework

This directory contains a comprehensive implementation of Federated Learning (FL) for the AiDotNet library, addressing Issue #398 (Phase 3).

## Overview

Federated Learning is a privacy-preserving distributed machine learning approach where multiple clients (devices, institutions, edge nodes) collaboratively train a shared model without sharing their raw data. Only model updates are exchanged, ensuring data privacy.

## Features Implemented

### Core Algorithms

#### 1. FedAvg (Federated Averaging)
- **Location**: `Aggregators/FedAvgAggregationStrategy.cs`
- **Description**: The foundational FL algorithm that performs weighted averaging of client model updates
- **Use Case**: Standard federated learning with IID or mildly non-IID data
- **Reference**: McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"

#### 2. FedProx (Federated Proximal)
- **Location**: `Aggregators/FedProxAggregationStrategy.cs`
- **Description**: Handles system heterogeneity by adding proximal terms to prevent client drift
- **Use Case**: Non-IID data, heterogeneous client capabilities, stragglers
- **Key Parameter**: μ (proximal term coefficient)
- **Reference**: Li et al. (2020) - "Federated Optimization in Heterogeneous Networks"

#### 3. FedBN (Federated Batch Normalization)
- **Location**: `Aggregators/FedBNAggregationStrategy.cs`
- **Description**: Keeps batch normalization layers local while aggregating other layers
- **Use Case**: Deep neural networks with batch normalization, non-IID data with distribution shift
- **Reference**: Li et al. (2021) - "Federated Learning on Non-IID Data Silos"

### Privacy Features

#### 1. Differential Privacy
- **Location**: `Privacy/GaussianDifferentialPrivacy.cs`
- **Description**: Implements (ε, δ)-differential privacy using the Gaussian mechanism
- **Features**:
  - Gradient clipping for sensitivity bounding
  - Calibrated Gaussian noise addition
  - Privacy budget tracking
  - Configurable ε (privacy budget) and δ (failure probability)
- **Reference**: Dwork & Roth (2014) - "The Algorithmic Foundations of Differential Privacy"

#### 2. Secure Aggregation
- **Location**: `Privacy/SecureAggregation.cs`
- **Description**: Cryptographic protocol to aggregate updates without revealing individual contributions
- **Features**:
  - Pairwise secret masking
  - Server only sees aggregated result
  - Protection against honest-but-curious server
- **Reference**: Bonawitz et al. (2017) - "Practical Secure Aggregation for Privacy-Preserving Machine Learning"

### Personalization

#### Personalized Federated Learning
- **Location**: `Personalization/PersonalizedFederatedLearning.cs`
- **Description**: Enables client-specific model layers while maintaining shared global layers
- **Features**:
  - Layer-wise personalization
  - Configurable personalization fraction
  - Model split statistics
  - Flexible personalization strategies
- **Use Case**: Non-IID data, client-specific adaptations, multi-task learning
- **Reference**: Fallah et al. (2020) - "Personalized Federated Learning: A Meta-Learning Approach"

## Architecture

### Interfaces

All core interfaces are located in `src/Interfaces/`:

- **IFederatedTrainer**: Main trainer for orchestrating federated learning
- **IAggregationStrategy**: Strategy pattern for different aggregation algorithms
- **IPrivacyMechanism**: Privacy-preserving mechanisms for protecting client data
- **IClientModel**: Client-side model operations

### Configuration

- **FederatedLearningOptions** (`src/Models/Options/FederatedLearningOptions.cs`): Comprehensive configuration options
  - Client management (number of clients, selection fraction)
  - Training parameters (learning rate, epochs, batch size)
  - Privacy settings (differential privacy, secure aggregation)
  - Convergence criteria
  - Personalization settings
  - Compression options

- **FederatedLearningMetadata** (`src/Models/FederatedLearningMetadata.cs`): Training metrics and statistics
  - Performance metrics (accuracy, loss)
  - Resource usage (time, communication)
  - Privacy budget tracking
  - Per-round detailed metrics

## Usage Examples

### Basic FedAvg

```csharp
using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.Models.Options;

// Create FedAvg aggregation strategy
var aggregator = new FedAvgAggregationStrategy<double>();

// Define client models and weights
var clientModels = new Dictionary<int, Dictionary<string, double[]>>
{
    [0] = new Dictionary<string, double[]> { ["layer1"] = new[] { 1.0, 2.0 } },
    [1] = new Dictionary<string, double[]> { ["layer1"] = new[] { 3.0, 4.0 } }
};

var clientWeights = new Dictionary<int, double>
{
    [0] = 100.0,  // 100 samples
    [1] = 200.0   // 200 samples
};

// Aggregate models
var globalModel = aggregator.Aggregate(clientModels, clientWeights);
```

### Differential Privacy

```csharp
using AiDotNet.FederatedLearning.Privacy;

// Create differential privacy mechanism
var dp = new GaussianDifferentialPrivacy<double>(
    clipNorm: 1.0,
    randomSeed: 42  // For reproducibility
);

// Apply privacy to model
var privateModel = dp.ApplyPrivacy(
    model: clientModel,
    epsilon: 1.0,    // Privacy budget
    delta: 1e-5      // Failure probability
);

// Check privacy budget consumed
Console.WriteLine($"Privacy budget used: {dp.GetPrivacyBudgetConsumed()}");
```

### Personalized Federated Learning

```csharp
using AiDotNet.FederatedLearning.Personalization;

// Create personalization handler
var pfl = new PersonalizedFederatedLearning<double>(
    personalizationFraction: 0.2  // Keep last 20% of layers personalized
);

// Identify which layers to personalize
pfl.IdentifyPersonalizedLayers(modelStructure, strategy: "last_n");

// Separate model into global and personalized parts
pfl.SeparateModel(
    fullModel: clientModel,
    out var globalPart,
    out var personalizedPart
);

// Send only globalPart to server for aggregation
// Keep personalizedPart on client

// After receiving aggregated global model from server
var updatedFullModel = pfl.CombineModels(
    globalUpdate: aggregatedGlobalModel,
    personalizedLayers: personalizedPart
);
```

## Configuration Options

Key parameters in `FederatedLearningOptions`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NumberOfClients` | int | 10 | Total number of participating clients |
| `ClientSelectionFraction` | double | 1.0 | Fraction of clients selected per round (0.0-1.0) |
| `LocalEpochs` | int | 5 | Number of epochs each client trains locally |
| `MaxRounds` | int | 100 | Maximum federated learning rounds |
| `LearningRate` | double | 0.01 | Learning rate for local training |
| `BatchSize` | int | 32 | Batch size for local training |
| `UseDifferentialPrivacy` | bool | false | Enable differential privacy |
| `PrivacyEpsilon` | double | 1.0 | Privacy budget (ε) |
| `PrivacyDelta` | double | 1e-5 | Privacy failure probability (δ) |
| `UseSecureAggregation` | bool | false | Enable secure aggregation |
| `AggregationStrategy` | string | "FedAvg" | Aggregation algorithm to use |
| `ProximalMu` | double | 0.01 | FedProx proximal term coefficient |
| `EnablePersonalization` | bool | false | Enable personalized layers |
| `PersonalizationLayerFraction` | double | 0.2 | Fraction of layers to personalize |

## Success Criteria (from Issue #398)

✅ **Core Algorithms**: FedAvg, FedProx, FedBN implemented
✅ **Privacy Features**: Differential Privacy and Secure Aggregation implemented
✅ **Personalization**: Personalized Federated Learning implemented
✅ **Architecture**: Clean interfaces (IFederatedTrainer, IAggregationStrategy)
✅ **Configuration**: Comprehensive options and metadata classes
✅ **Documentation**: Extensive XML documentation with beginner-friendly explanations
✅ **Testing**: Unit tests for core algorithms and privacy mechanisms

## Testing

Unit tests are located in `tests/AiDotNet.Tests/FederatedLearning/`:

- `FedAvgAggregationStrategyTests.cs`: Tests for FedAvg aggregation
- `GaussianDifferentialPrivacyTests.cs`: Tests for differential privacy mechanism

Run tests:
```bash
dotnet test tests/AiDotNet.Tests/
```

## Future Enhancements

Potential extensions for future phases:

1. **LEAF Dataset Adapters**: Dataset-specific adapters for the full LEAF suite (FEMNIST, Sent140, Shakespeare, CelebA, etc.)
2. **Communication Efficiency**: Implement gradient compression and quantization
3. **Advanced Privacy**: Add Rényi Differential Privacy for tighter composition bounds
4. **Byzantine Robustness**: Implement Krum, Median, Trimmed Mean aggregation
5. **Meta-Learning**: Add MAML (Model-Agnostic Meta-Learning) for personalization
6. **Asynchronous FL**: Support for asynchronous client updates
7. **Vertical FL**: Support for vertically partitioned data
8. **Cross-Silo FL**: Enhanced support for enterprise federated learning scenarios

## References

1. McMahan, H. B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
2. Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys 2020.
3. Li, X., et al. (2021). "Federated Learning on Non-IID Data Silos: An Experimental Study." ICDE 2021.
4. Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning." CCS 2017.
5. Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy." Foundations and Trends in Theoretical Computer Science.
6. Fallah, A., et al. (2020). "Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach." NeurIPS 2020.

## Contributing

When adding new federated learning algorithms or features:

1. Follow the existing interface patterns (IAggregationStrategy, IPrivacyMechanism, etc.)
2. Add comprehensive XML documentation with beginner-friendly explanations
3. Include mathematical formulations and references
4. Add unit tests for new functionality
5. Update this README with usage examples

## License

This implementation is part of the AiDotNet library and is licensed under Apache-2.0.
