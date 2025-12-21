# ML Training Infrastructure and Experiment Tracking

This document describes the comprehensive ML training infrastructure added to AiDotNet, providing enterprise-grade capabilities for experiment tracking, hyperparameter optimization, checkpoint management, training monitoring, model registry, and data versioning.

## Table of Contents

1. [Overview](#overview)
2. [Experiment Tracking](#experiment-tracking)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Checkpoint Management](#checkpoint-management)
5. [Training Monitoring](#training-monitoring)
6. [Model Registry](#model-registry)
7. [Data Version Control](#data-version-control)
8. [Quick Start Examples](#quick-start-examples)

---

## Overview

The ML Training Infrastructure provides feature parity with industry-standard tools like MLflow, Optuna, and DVC, while maintaining the type-safe, well-documented approach of AiDotNet.

### Key Components

| Component | Purpose | Similar To |
|-----------|---------|------------|
| **Experiment Tracking** | Log and compare training runs | MLflow Tracking |
| **Hyperparameter Optimization** | Automated hyperparameter search | Optuna |
| **Checkpoint Management** | Save/restore training state | PyTorch checkpoints |
| **Training Monitoring** | Real-time training visualization | TensorBoard |
| **Model Registry** | Centralized model storage | MLflow Model Registry |
| **Data Version Control** | Track dataset versions | DVC |

---

## Experiment Tracking

Track machine learning experiments, log parameters and metrics, and compare different runs.

### Key Interfaces

- `IExperimentTracker<T>` - Main tracker interface
- `IExperiment` - Represents an experiment
- `IExperimentRun<T>` - Represents a training run

### Implementation

The `ExperimentTracker<T>` class provides a complete implementation with file-based storage.

### Basic Usage

```csharp
using AiDotNet.ExperimentTracking;

// Create experiment tracker
var tracker = new ExperimentTracker<double>();

// Create an experiment
var expId = tracker.CreateExperiment(
    "house-price-prediction",
    "Predicting house prices using various algorithms"
);

// Start a training run
var run = tracker.StartRun(expId, "neural-network-v1");

// Log hyperparameters
run.LogParameter("learning_rate", 0.01);
run.LogParameter("batch_size", 32);
run.LogParameter("epochs", 100);

// Log metrics during training
for (int epoch = 0; epoch < 100; epoch++)
{
    double loss = TrainEpoch(); // Your training logic
    double accuracy = Evaluate(); // Your evaluation logic

    run.LogMetric("loss", loss, epoch);
    run.LogMetric("accuracy", accuracy, epoch);
}

// Log trained model as artifact
run.LogModel(trainedModel);

// Mark run as complete
run.Complete();
```

### Advanced Features

**Searching Runs**
```csharp
// Search across all experiments
var bestRuns = tracker.SearchRuns("accuracy > 0.9", maxResults: 10);

// List runs in an experiment
var runs = tracker.ListRuns(expId);
```

**Comparing Runs**
```csharp
// Get all runs and compare metrics
var allRuns = tracker.ListRuns(expId);
foreach (var run in allRuns)
{
    var metrics = run.GetMetrics();
    var finalAccuracy = run.GetLatestMetric("accuracy");
    Console.WriteLine($"Run {run.RunName}: {finalAccuracy}");
}
```

### Storage

By default, experiments are stored in `./mlruns/` with the following structure:

```
mlruns/
├── <experiment-id>/
│   ├── meta.json              # Experiment metadata
│   ├── <run-id-1>/
│   │   ├── meta.json          # Run metadata
│   │   └── artifacts/         # Logged artifacts
│   └── <run-id-2>/
│       └── ...
```

---

## Hyperparameter Optimization

Automated hyperparameter search using various strategies.

### Key Interfaces

- `IHyperparameterOptimizer<T, TInput, TOutput>` - Main optimizer interface

### Implementations

1. **RandomSearchOptimizer** - Random sampling (fast, effective)
2. **GridSearchOptimizer** - Exhaustive grid search (thorough, slow)

### Basic Usage

```csharp
using AiDotNet.HyperparameterOptimization;
using AiDotNet.Models;

// Define search space
var searchSpace = new HyperparameterSearchSpace();
searchSpace.AddContinuous("learning_rate", 0.0001, 0.1, logScale: true);
searchSpace.AddInteger("batch_size", 16, 128, step: 16);
searchSpace.AddCategorical("optimizer", "adam", "sgd", "rmsprop");
searchSpace.AddInteger("hidden_units", 32, 512, step: 32);

// Define objective function
Func<Dictionary<string, object>, double> objectiveFunction = (parameters) =>
{
    // Train model with these parameters
    var model = TrainModel(parameters);

    // Return validation accuracy (or negative loss for minimization)
    return EvaluateModel(model);
};

// Run optimization
var optimizer = new RandomSearchOptimizer<double, Matrix<double>, Vector<double>>(
    maximize: true,  // Maximize accuracy
    seed: 42         // For reproducibility
);

var result = optimizer.Optimize(objectiveFunction, searchSpace, nTrials: 50);

// Get best parameters
Console.WriteLine($"Best accuracy: {result.BestObjectiveValue}");
foreach (var param in result.BestParameters)
{
    Console.WriteLine($"{param.Key}: {param.Value}");
}

// Analyze optimization history
var history = result.GetOptimizationHistory();
foreach (var (trial, value) in history)
{
    Console.WriteLine($"Trial {trial}: {value}");
}
```

### Search Space Types

**Continuous Parameters** (decimals)
```csharp
// Linear scale: 0.0 to 1.0
searchSpace.AddContinuous("dropout_rate", 0.0, 1.0);

// Log scale: for learning rates
searchSpace.AddContinuous("learning_rate", 1e-5, 1e-1, logScale: true);
```

**Integer Parameters**
```csharp
// With step size
searchSpace.AddInteger("num_layers", 1, 10, step: 1);
searchSpace.AddInteger("batch_size", 16, 256, step: 16);
```

**Categorical Parameters** (discrete choices)
```csharp
searchSpace.AddCategorical("activation", "relu", "tanh", "sigmoid");
searchSpace.AddCategorical("optimizer_type", "adam", "sgd", "rmsprop");
```

**Boolean Parameters**
```csharp
searchSpace.AddBoolean("use_batch_norm");
searchSpace.AddBoolean("use_dropout");
```

### Grid Search vs Random Search

**Use Grid Search when:**
- You have a small search space (few parameters, few values each)
- You want to try every combination
- You need reproducible, systematic exploration

**Use Random Search when:**
- You have a large search space
- You want faster results
- Some hyperparameters are more important than others
- You want better coverage of continuous parameters

---

## Checkpoint Management

Save and restore training state to enable resumption and track best models.

### Key Interfaces

- `ICheckpointManager<T, TInput, TOutput>` - Checkpoint management interface

### Implementation

The `CheckpointManager<T, TInput, TOutput>` class provides complete checkpoint functionality.

### Basic Usage

```csharp
using AiDotNet.CheckpointManagement;

// Create checkpoint manager
var checkpointManager = new CheckpointManager<double, Matrix<double>, Vector<double>>();

// During training, save checkpoints
for (int epoch = 0; epoch < 100; epoch++)
{
    TrainEpoch(model, optimizer);

    var metrics = new Dictionary<string, double>
    {
        ["loss"] = ComputeLoss(),
        ["accuracy"] = ComputeAccuracy()
    };

    // Save checkpoint every 10 epochs
    if (epoch % 10 == 0)
    {
        var checkpointId = checkpointManager.SaveCheckpoint(
            model,
            optimizer,
            epoch,
            step: epoch * stepsPerEpoch,
            metrics
        );
        Console.WriteLine($"Saved checkpoint: {checkpointId}");
    }
}

// Resume training from latest checkpoint
var checkpoint = checkpointManager.LoadLatestCheckpoint();
if (checkpoint != null)
{
    model = checkpoint.Model;
    optimizer = checkpoint.Optimizer;
    int resumeEpoch = checkpoint.Epoch;

    Console.WriteLine($"Resuming from epoch {resumeEpoch}");
}

// Load best checkpoint by metric
var bestCheckpoint = checkpointManager.LoadBestCheckpoint(
    "accuracy",
    MetricOptimizationDirection.Maximize
);
```

### Auto-Checkpointing

```csharp
// Configure automatic checkpointing
checkpointManager.ConfigureAutoCheckpointing(
    saveFrequency: 5,          // Save every 5 steps
    keepLast: 10,              // Keep 10 most recent
    saveOnImprovement: true,   // Save when metric improves
    metricName: "accuracy"     // Track this metric
);
```

### Checkpoint Cleanup

```csharp
// Keep only last N checkpoints
int deleted = checkpointManager.CleanupOldCheckpoints(keepLast: 5);

// Keep only best N checkpoints by metric
int deleted = checkpointManager.CleanupKeepBest(
    "accuracy",
    keepBest: 3,
    direction: MetricOptimizationDirection.Maximize
);
```

### Listing Checkpoints

```csharp
// List all checkpoints sorted by creation time
var checkpoints = checkpointManager.ListCheckpoints(
    sortBy: "created",
    descending: true
);

foreach (var cp in checkpoints)
{
    Console.WriteLine($"Checkpoint {cp.CheckpointId}:");
    Console.WriteLine($"  Epoch: {cp.Epoch}, Step: {cp.Step}");
    Console.WriteLine($"  Created: {cp.CreatedAt}");
    Console.WriteLine($"  Metrics: {string.Join(", ", cp.Metrics.Select(m => $"{m.Key}={m.Value}"))}");
}
```

---

## Training Monitoring

Real-time monitoring of training progress, metrics, and system resources.

### Key Interfaces

- `ITrainingMonitor<T>` - Training monitoring interface
- `TrainingSpeedStats` - Speed and progress statistics
- `ResourceUsageStats` - System resource usage

### Basic Usage

```csharp
using AiDotNet.Interfaces;

// Create training monitor
ITrainingMonitor<double> monitor = new TrainingMonitor<double>();

// Start monitoring session
var sessionId = monitor.StartSession("experiment-1", new Dictionary<string, object>
{
    ["model_type"] = "neural_network",
    ["dataset"] = "mnist"
});

// During training
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    monitor.OnEpochStart(sessionId, epoch);

    for (int step = 0; step < stepsPerEpoch; step++)
    {
        // Training step
        var (loss, accuracy) = TrainStep();

        // Log metrics
        monitor.LogMetrics(sessionId, new Dictionary<string, double>
        {
            ["loss"] = loss,
            ["accuracy"] = accuracy
        }, step);

        // Log resource usage
        monitor.LogResourceUsage(
            sessionId,
            cpuUsage: GetCpuUsage(),
            memoryUsage: GetMemoryUsage()
        );

        // Update progress
        monitor.UpdateProgress(
            sessionId,
            currentStep: step,
            totalSteps: stepsPerEpoch,
            currentEpoch: epoch,
            totalEpochs: numEpochs
        );
    }

    monitor.OnEpochEnd(sessionId, epoch, epochMetrics, duration);
}

// Check for issues
var issues = monitor.CheckForIssues(sessionId);
foreach (var issue in issues)
{
    Console.WriteLine($"Warning: {issue}");
}

// Get statistics
var speedStats = monitor.GetSpeedStats(sessionId);
Console.WriteLine($"Speed: {speedStats.IterationsPerSecond:F2} it/s");
Console.WriteLine($"ETA: {speedStats.EstimatedTimeRemaining}");

// End session
monitor.EndSession(sessionId);
```

### Logging Messages

```csharp
// Log informational message
monitor.LogMessage(sessionId, LogLevel.Info, "Starting validation phase");

// Log warning
monitor.LogMessage(sessionId, LogLevel.Warning, "High memory usage detected");

// Log error
monitor.LogMessage(sessionId, LogLevel.Error, "Training diverged - NaN loss");
```

### Exporting and Visualization

```csharp
// Export training data
monitor.ExportData(sessionId, "training_log.json", format: "json");

// Create visualization
monitor.CreateVisualization(
    sessionId,
    new List<string> { "loss", "accuracy" },
    "training_curves.png"
);
```

---

## Model Registry

Centralized repository for managing trained models with versioning and lifecycle management.

### Key Interfaces

- `IModelRegistry<T, TInput, TOutput>` - Model registry interface
- `ModelStage` - Lifecycle stages (Development, Staging, Production, Archived)

### Basic Usage

```csharp
using AiDotNet.Interfaces;

// Create model registry
IModelRegistry<double, Matrix<double>, Vector<double>> registry =
    new ModelRegistry<double, Matrix<double>, Vector<double>>();

// Register a new model
var modelId = registry.RegisterModel(
    name: "house-price-predictor",
    model: trainedModel,
    metadata: modelMetadata,
    tags: new Dictionary<string, string>
    {
        ["framework"] = "aidotnet",
        ["task"] = "regression"
    }
);

// Create new versions
int v2 = registry.CreateModelVersion(
    "house-price-predictor",
    improvedModel,
    newMetadata,
    description: "Added feature engineering"
);

// Get model versions
var latestModel = registry.GetLatestModel("house-price-predictor");
var specificVersion = registry.GetModel("house-price-predictor", version: 2);

// Promote model through stages
registry.TransitionModelStage(
    "house-price-predictor",
    version: 2,
    ModelStage.Staging
);

// After testing in staging
registry.TransitionModelStage(
    "house-price-predictor",
    version: 2,
    ModelStage.Production,
    archivePrevious: true  // Archive previous production model
);

// Get production model
var productionModel = registry.GetModelByStage(
    "house-price-predictor",
    ModelStage.Production
);
```

### Searching Models

```csharp
// Search by criteria
var searchCriteria = new ModelSearchCriteria<double>
{
    NamePattern = "*predictor*",
    Tags = new Dictionary<string, string> { ["task"] = "regression" },
    MinMetricValue = 0.9,
    MetricName = "accuracy"
};

var models = registry.SearchModels(searchCriteria);
```

### Comparing Models

```csharp
// Compare two versions
var comparison = registry.CompareModels(
    "house-price-predictor",
    version1: 1,
    version2: 2
);

Console.WriteLine($"Version 1 accuracy: {comparison.Version1Metrics["accuracy"]}");
Console.WriteLine($"Version 2 accuracy: {comparison.Version2Metrics["accuracy"]}");
Console.WriteLine($"Improvement: {comparison.MetricDifferences["accuracy"]}");
```

### Model Lineage

```csharp
// Track model lineage (what data, what experiment)
var lineage = registry.GetModelLineage("house-price-predictor", version: 2);

Console.WriteLine($"Trained from experiment: {lineage.ExperimentId}");
Console.WriteLine($"Training data: {lineage.DatasetVersion}");
Console.WriteLine($"Parent models: {string.Join(", ", lineage.ParentModels)}");
```

---

## Data Version Control

Track dataset versions to ensure reproducibility and traceability.

### Key Interfaces

- `IDataVersionControl<T>` - Data versioning interface

### Basic Usage

```csharp
using AiDotNet.Interfaces;

// Create data version control
IDataVersionControl<double> dvc = new DataVersionControl<double>();

// Create dataset version
var versionHash = dvc.CreateDatasetVersion(
    datasetName: "housing-data",
    dataPath: "/data/housing.csv",
    description: "Initial dataset with 10,000 samples",
    metadata: new Dictionary<string, object>
    {
        ["rows"] = 10000,
        ["columns"] = 15,
        ["date_collected"] = DateTime.UtcNow
    }
);

// Tag important versions
dvc.TagDatasetVersion("housing-data", versionHash, "production-v1");

// Link dataset to training run
dvc.LinkDatasetToRun(
    "housing-data",
    versionHash,
    runId: experimentRunId,
    modelId: trainedModelId
);

// Later: reproduce exact training conditions
var datasetVersion = dvc.GetDatasetByTag("housing-data", "production-v1");
var dataPath = datasetVersion.DataPath;

// Verify data integrity
bool isValid = dvc.VerifyDatasetIntegrity(
    "housing-data",
    versionHash,
    currentDataPath: dataPath
);

if (!isValid)
{
    Console.WriteLine("Warning: Data has been modified!");
}
```

### Dataset Lineage

```csharp
// Record how dataset was created
var lineage = new DatasetLineage
{
    SourceDatasets = new List<string> { "raw-data-v1" },
    TransformationsApplied = new List<string>
    {
        "remove_duplicates",
        "handle_missing_values",
        "normalize_features"
    },
    Code = "preprocessing_pipeline.py",
    CreatedBy = "data-scientist@company.com"
};

dvc.RecordDatasetLineage("housing-data", versionHash, lineage);
```

### Dataset Snapshots

```csharp
// Create snapshot of multiple related datasets
var snapshotId = dvc.CreateDatasetSnapshot(
    "experiment-2024-01",
    new Dictionary<string, string>
    {
        ["train"] = trainDataHash,
        ["validation"] = valDataHash,
        ["test"] = testDataHash
    },
    description: "Complete dataset split for January 2024 experiments"
);

// Later: restore entire snapshot
var snapshot = dvc.GetDatasetSnapshot("experiment-2024-01");
var trainData = dvc.GetDatasetVersion("train", snapshot.Datasets["train"]);
var valData = dvc.GetDatasetVersion("validation", snapshot.Datasets["validation"]);
var testData = dvc.GetDatasetVersion("test", snapshot.Datasets["test"]);
```

---

## Quick Start Examples

### Complete Training Pipeline with All Components

```csharp
using AiDotNet.ExperimentTracking;
using AiDotNet.HyperparameterOptimization;
using AiDotNet.CheckpointManagement;

// 1. Setup infrastructure
var tracker = new ExperimentTracker<double>();
var checkpointMgr = new CheckpointManager<double, Matrix<double>, Vector<double>>();
var dvc = new DataVersionControl<double>();

// 2. Version your data
var dataHash = dvc.CreateDatasetVersion(
    "mnist",
    "/data/mnist.csv",
    "MNIST handwritten digits dataset"
);

// 3. Create experiment
var expId = tracker.CreateExperiment(
    "mnist-classification",
    "Digit classification experiments"
);

// 4. Hyperparameter optimization
var searchSpace = new HyperparameterSearchSpace();
searchSpace.AddContinuous("learning_rate", 1e-4, 1e-1, logScale: true);
searchSpace.AddInteger("batch_size", 16, 128, step: 16);
searchSpace.AddInteger("hidden_size", 64, 512, step: 64);

Func<Dictionary<string, object>, double> objective = (params) =>
{
    var run = tracker.StartRun(expId);
    run.LogParameters(params);

    // Train model
    var model = TrainModel(params);
    var accuracy = EvaluateModel(model);

    run.LogMetric("accuracy", accuracy);
    run.Complete();

    return accuracy;
};

var optimizer = new RandomSearchOptimizer<double, Matrix<double>, Vector<double>>(
    maximize: true,
    seed: 42
);

var result = optimizer.Optimize(objective, searchSpace, nTrials: 20);

// 5. Train final model with best hyperparameters
var finalRun = tracker.StartRun(expId, "final-model");
finalRun.LogParameters(result.BestParameters);

// Link dataset
dvc.LinkDatasetToRun("mnist", dataHash, finalRun.RunId);

var finalModel = TrainModelWithCheckpoints(
    result.BestParameters,
    checkpointMgr,
    finalRun
);

// 6. Save final model
finalRun.LogModel(finalModel);
finalRun.Complete();

Console.WriteLine($"Best accuracy: {result.BestObjectiveValue:F4}");
Console.WriteLine($"Experiment ID: {expId}");
Console.WriteLine($"Run ID: {finalRun.RunId}");
```

---

## Architecture and Design

### Thread Safety

All implementations are thread-safe and can be used concurrently:
- Internal locking ensures consistency
- File operations are atomic where possible
- Readers don't block other readers

### Storage

- **Experiment Tracking**: File-based storage in `./mlruns/`
- **Checkpoints**: Binary serialization in `./checkpoints/`
- **Models**: Registry storage in `./model_registry/`
- **Data Versions**: Metadata in `./dvc/`

All storage locations are configurable via constructor parameters.

### Performance Considerations

- Checkpoints are saved asynchronously where possible
- Metrics are batched for efficient logging
- File I/O uses buffering for large datasets
- Memory usage is optimized for large training runs

---

## Comparison with Other Tools

| Feature | AiDotNet | MLflow | Optuna | DVC |
|---------|----------|--------|--------|-----|
| Type Safety | ✅ Full | ❌ Python | ❌ Python | ❌ Python |
| .NET Native | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Experiment Tracking | ✅ Yes | ✅ Yes | ⚠️ Limited | ❌ No |
| Hyperparameter Opt | ✅ Yes | ⚠️ Limited | ✅ Yes | ❌ No |
| Model Registry | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| Data Versioning | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| Checkpointing | ✅ Yes | ⚠️ Limited | ❌ No | ❌ No |
| Documentation | ✅ Excellent | ✅ Good | ✅ Good | ✅ Good |

---

## Future Enhancements

Planned for future releases:

1. **Distributed Training Support**
   - Multi-GPU checkpoint synchronization
   - Distributed hyperparameter optimization

2. **Advanced Optimizers**
   - Bayesian optimization
   - Hyperband/ASHA for early stopping
   - Multi-objective optimization

3. **Visualization**
   - Web UI for experiment comparison
   - Interactive training dashboards
   - Hyperparameter importance plots

4. **Integration**
   - Cloud storage backends (Azure, AWS, GCP)
   - Database backends for metadata
   - CI/CD pipeline integration

---

## Support

For questions, issues, or feature requests, please visit:
- GitHub Issues: https://github.com/ooples/AiDotNet/issues
- Documentation: https://github.com/ooples/AiDotNet

## License

This infrastructure is part of AiDotNet and follows the same license as the main project.
