# Issue #415: Junior Developer Implementation Guide
## ML Training Infrastructure and Experiment Tracking

---

## Table of Contents
1. [Understanding ML Training Infrastructure](#understanding-ml-training-infrastructure)
2. [Understanding Experiment Tracking](#understanding-experiment-tracking)
3. [Understanding Hyperparameter Optimization](#understanding-hyperparameter-optimization)
4. [Understanding Model Registry](#understanding-model-registry)
5. [Architecture Overview](#architecture-overview)
6. [Phase 1: Core Training Infrastructure](#phase-1-core-training-infrastructure)
7. [Phase 2: Experiment Tracking](#phase-2-experiment-tracking)
8. [Phase 3: Hyperparameter Optimization](#phase-3-hyperparameter-optimization)
9. [Phase 4: Model Registry](#phase-4-model-registry)
10. [Testing Strategy](#testing-strategy)
11. [Common Pitfalls](#common-pitfalls)

---

## Understanding ML Training Infrastructure

### What Is ML Training Infrastructure?

ML Training Infrastructure is the **foundational layer** that manages the training process of machine learning models. It provides:

1. **Training Loop Management**: Orchestrates epochs, batches, forward/backward passes
2. **Metric Tracking**: Records loss, accuracy, and custom metrics during training
3. **Checkpoint Management**: Saves model states for recovery and evaluation
4. **Resource Management**: Handles memory, GPU/CPU utilization
5. **Logging and Visualization**: Reports progress and training curves

### Why Do We Need It?

**Problem**: Training models involves repetitive boilerplate code:
```csharp
// Without infrastructure (manual, error-prone)
for (int epoch = 0; epoch < 100; epoch++)
{
    foreach (var batch in dataLoader)
    {
        var output = model.Forward(batch.Input);
        var loss = lossFunction.Compute(output, batch.Target);
        var gradients = loss.Backward();
        optimizer.Step(gradients);

        // Manual logging, checkpointing, metric calculation...
    }
}
```

**Solution**: Infrastructure automates and standardizes training:
```csharp
// With infrastructure (clean, consistent)
var trainer = new Trainer<TInput, TOutput>(model, optimizer, lossFunction);
trainer.OnEpochEnd += (metrics) => Console.WriteLine($"Epoch {metrics.Epoch}: Loss={metrics.Loss}");
trainer.Train(trainDataLoader, epochs: 100, validationDataLoader);
```

### Key Concepts

#### 1. Training Loop
```
For each epoch:
  For each batch in training data:
    1. Forward pass: predictions = model(inputs)
    2. Loss computation: loss = loss_function(predictions, targets)
    3. Backward pass: gradients = loss.backward()
    4. Optimizer step: update model weights
    5. Record metrics: accuracy, loss, etc.

  Validation phase:
    Evaluate model on validation set
    Record validation metrics

  Checkpoint:
    Save model if validation improved
```

#### 2. Metrics
- **Training Metrics**: Computed during training (loss, accuracy per batch)
- **Validation Metrics**: Computed after each epoch on validation set
- **Aggregation**: Metrics averaged or accumulated across batches/epochs

#### 3. Callbacks
Allow custom logic at specific training stages:
- `OnEpochStart`: Before each epoch begins
- `OnEpochEnd`: After each epoch completes
- `OnBatchEnd`: After each batch processes
- `OnTrainingComplete`: When training finishes

---

## Understanding Experiment Tracking

### What Is Experiment Tracking?

Experiment tracking is a **systematic way to record and compare** different training runs. Think of it like a lab notebook for ML experiments.

### Why Track Experiments?

**Problem**: Without tracking, you lose critical information:
- "Which hyperparameters gave the best result?"
- "What learning rate did we use in that successful run?"
- "Can we reproduce last week's model?"

**Solution**: Track everything automatically:
```csharp
var experiment = tracker.StartExperiment("mnist_classifier");
experiment.LogParameter("learning_rate", 0.001);
experiment.LogParameter("batch_size", 32);
experiment.LogMetric("train_loss", 0.45, step: 100);
experiment.LogMetric("val_accuracy", 0.92, step: 100);
experiment.LogArtifact("model.pt", modelBytes);
```

### MLflow-Like Architecture

**MLflow** is the industry standard for experiment tracking. We'll implement similar concepts:

#### 1. Experiment
A logical grouping of related runs (e.g., "ResNet on ImageNet"):
```csharp
public class Experiment
{
    public string ExperimentId { get; set; }
    public string Name { get; set; }
    public DateTime CreatedAt { get; set; }
    public List<Run> Runs { get; set; }
}
```

#### 2. Run
A single training execution with specific parameters:
```csharp
public class Run
{
    public string RunId { get; set; }
    public string ExperimentId { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public RunStatus Status { get; set; } // Running, Finished, Failed
    public Dictionary<string, object> Parameters { get; set; }
    public Dictionary<string, List<MetricEntry>> Metrics { get; set; }
    public Dictionary<string, string> Tags { get; set; }
}
```

#### 3. Metrics
Time-series data recorded during training:
```csharp
public class MetricEntry
{
    public double Value { get; set; }
    public long Step { get; set; }    // Batch number or epoch
    public DateTime Timestamp { get; set; }
}
```

#### 4. Artifacts
Files produced by training (models, plots, configs):
```csharp
public class Artifact
{
    public string Name { get; set; }
    public string Path { get; set; }
    public long SizeBytes { get; set; }
    public string ContentType { get; set; }
}
```

### Storage Backend

Experiments need persistent storage:
```
experiments/
├── experiment_001/
│   ├── meta.json          # Experiment metadata
│   ├── run_001/
│   │   ├── params.json    # Hyperparameters
│   │   ├── metrics.json   # Training metrics
│   │   ├── tags.json      # User tags
│   │   └── artifacts/
│   │       ├── model_epoch_10.pt
│   │       └── training_curve.png
│   └── run_002/
│       └── ...
```

---

## Understanding Hyperparameter Optimization

### What Is Hyperparameter Optimization?

Hyperparameter optimization (HPO) is the process of **automatically finding the best hyperparameters** for a model.

### Hyperparameters vs Parameters

**Parameters**: Learned during training (weights, biases)
**Hyperparameters**: Set before training (learning rate, batch size, number of layers)

### Why Automate HPO?

**Manual tuning is inefficient:**
```
Try learning_rate = 0.1 → Accuracy = 0.75
Try learning_rate = 0.01 → Accuracy = 0.88
Try learning_rate = 0.001 → Accuracy = 0.91
Try learning_rate = 0.0001 → Accuracy = 0.85
```

**Automated HPO explores systematically:**
```csharp
var searchSpace = new SearchSpace
{
    { "learning_rate", new ContinuousSpace(1e-5, 1e-1, logScale: true) },
    { "batch_size", new DiscreteSpace(16, 32, 64, 128) },
    { "num_layers", new IntegerSpace(1, 5) }
};

var optimizer = new TPEOptimizer(searchSpace, trials: 50);
var bestParams = optimizer.Optimize(objectiveFunction);
```

### Optuna-Like Architecture

**Optuna** is a popular HPO framework. We'll implement similar algorithms:

#### 1. Search Spaces
Define the range of possible values:
```csharp
public abstract class SearchSpace
{
    public abstract object Sample(Random rng);
}

public class ContinuousSpace : SearchSpace
{
    public double Low { get; set; }
    public double High { get; set; }
    public bool LogScale { get; set; }  // For learning rates

    public override object Sample(Random rng)
    {
        double value = rng.NextDouble() * (High - Low) + Low;
        return LogScale ? Math.Exp(value) : value;
    }
}

public class DiscreteSpace : SearchSpace
{
    public object[] Choices { get; set; }

    public override object Sample(Random rng)
    {
        return Choices[rng.Next(Choices.Length)];
    }
}
```

#### 2. Trial
A single training run with specific hyperparameters:
```csharp
public class Trial
{
    public int TrialId { get; set; }
    public Dictionary<string, object> Parameters { get; set; }
    public double? ObjectiveValue { get; set; }  // e.g., validation accuracy
    public TrialState State { get; set; }  // Running, Complete, Failed, Pruned
}
```

#### 3. Sampling Algorithms

**Random Search**: Sample parameters randomly
```csharp
public class RandomSampler : ISampler
{
    public Dictionary<string, object> Sample(SearchSpace space, List<Trial> history)
    {
        var parameters = new Dictionary<string, object>();
        foreach (var param in space)
        {
            parameters[param.Key] = param.Value.Sample(_rng);
        }
        return parameters;
    }
}
```

**Tree-structured Parzen Estimator (TPE)**: Model promising regions
```csharp
public class TPESampler : ISampler
{
    public Dictionary<string, object> Sample(SearchSpace space, List<Trial> history)
    {
        // 1. Split trials into "good" (top 20%) and "bad" (bottom 80%)
        var sortedTrials = history.OrderByDescending(t => t.ObjectiveValue).ToList();
        var splitIndex = (int)(sortedTrials.Count * 0.2);
        var goodTrials = sortedTrials.Take(splitIndex).ToList();
        var badTrials = sortedTrials.Skip(splitIndex).ToList();

        // 2. Fit probability distributions to good and bad trials
        var goodDist = FitDistribution(goodTrials);
        var badDist = FitDistribution(badTrials);

        // 3. Sample from region where good/bad ratio is high
        return SampleFromRatio(goodDist, badDist, space);
    }
}
```

#### 4. Pruning (Early Stopping)

Stop unpromising trials early to save compute:
```csharp
public class MedianPruner : IPruner
{
    public bool ShouldPrune(Trial trial, int step, double intermediateValue)
    {
        // Get median performance of other trials at this step
        var medianValue = _history
            .Where(t => t.IntermediateValues.ContainsKey(step))
            .Select(t => t.IntermediateValues[step])
            .Median();

        // Prune if significantly worse than median
        return intermediateValue < medianValue * 0.8;
    }
}
```

---

## Understanding Model Registry

### What Is a Model Registry?

A model registry is a **centralized repository** for trained models, providing:
1. **Versioning**: Track model versions over time
2. **Metadata**: Store training info, performance metrics
3. **Lifecycle Management**: Track model stages (development, staging, production)
4. **Reproducibility**: Link models to training runs and code

### Why Do We Need It?

**Problem without registry:**
- Models scattered across filesystems: `model_v1_final.pt`, `model_v2_FINAL_FINAL.pt`
- No way to know which model is in production
- Can't reproduce model training
- No audit trail of model changes

**Solution with registry:**
```csharp
// Register new model
var modelVersion = registry.RegisterModel(
    name: "fraud_detector",
    modelBytes: modelBytes,
    sourceRun: run,
    metadata: new { accuracy = 0.95, f1_score = 0.92 }
);

// Promote to production
registry.SetModelStage(modelVersion, ModelStage.Production);

// Load production model
var productionModel = registry.GetProductionModel("fraud_detector");
```

### Model Lifecycle

```
Development → Staging → Production → Archived
     ↓           ↓          ↓            ↓
   Testing   Validation  Serving    Deprecated
```

### Registry Schema

```csharp
public class RegisteredModel
{
    public string ModelId { get; set; }
    public string Name { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? LastUpdated { get; set; }
    public string Description { get; set; }
    public List<ModelVersion> Versions { get; set; }
}

public class ModelVersion
{
    public string VersionId { get; set; }
    public string ModelId { get; set; }
    public int Version { get; set; }  // Incremental version number
    public string RunId { get; set; }  // Link to training run
    public ModelStage Stage { get; set; }
    public DateTime CreatedAt { get; set; }
    public string StoragePath { get; set; }
    public Dictionary<string, object> Metadata { get; set; }
    public Dictionary<string, double> Metrics { get; set; }
}

public enum ModelStage
{
    Development,
    Staging,
    Production,
    Archived
}
```

---

## Architecture Overview

### Component Relationships

```
┌─────────────────────────────────────────────────────────┐
│                   User Application                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Trainer<T>                           │
│  - Train() method                                       │
│  - Callbacks (OnEpochEnd, OnBatchEnd)                  │
│  - Checkpoint management                                │
└─────────────────────────────────────────────────────────┘
         │                  │                  │
         ↓                  ↓                  ↓
┌─────────────┐   ┌──────────────────┐   ┌─────────────┐
│  Experiment │   │  Hyperparameter  │   │    Model    │
│   Tracker   │   │    Optimizer     │   │   Registry  │
└─────────────┘   └──────────────────┘   └─────────────┘
         │                  │                  │
         ↓                  ↓                  ↓
┌─────────────────────────────────────────────────────────┐
│              Storage Backend (File System)              │
│  - Experiments                                          │
│  - Runs and metrics                                     │
│  - Models and artifacts                                 │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/Training/
├── Core/
│   ├── Trainer.cs                    # Main training orchestrator
│   ├── TrainingConfig.cs             # Training configuration
│   ├── Callbacks/
│   │   ├── ICallback.cs              # Callback interface
│   │   ├── EarlyStopping.cs          # Stop training when metric plateaus
│   │   ├── ReduceLROnPlateau.cs      # Reduce learning rate
│   │   └── ProgressCallback.cs       # Display progress bar
│   └── Metrics/
│       ├── IMetric.cs                # Metric interface
│       ├── Accuracy.cs               # Classification accuracy
│       ├── MeanSquaredError.cs       # Regression metric
│       └── MetricAggregator.cs       # Aggregate metrics over batches
│
├── ExperimentTracking/
│   ├── IExperimentTracker.cs         # Tracker interface
│   ├── FileSystemTracker.cs          # File-based implementation
│   ├── Models/
│   │   ├── Experiment.cs
│   │   ├── Run.cs
│   │   ├── MetricEntry.cs
│   │   └── Artifact.cs
│   └── Storage/
│       ├── IStorage.cs               # Storage abstraction
│       └── LocalFileStorage.cs       # Local disk storage
│
├── HyperparameterOptimization/
│   ├── IOptimizer.cs                 # HPO optimizer interface
│   ├── RandomSearchOptimizer.cs      # Random search
│   ├── TPEOptimizer.cs               # Tree-structured Parzen Estimator
│   ├── GridSearchOptimizer.cs        # Grid search
│   ├── SearchSpace/
│   │   ├── SearchSpace.cs            # Base class
│   │   ├── ContinuousSpace.cs        # Continuous range
│   │   ├── DiscreteSpace.cs          # Categorical choices
│   │   └── IntegerSpace.cs           # Integer range
│   ├── Pruning/
│   │   ├── IPruner.cs                # Pruning interface
│   │   ├── MedianPruner.cs           # Median-based pruning
│   │   └── SuccessiveHalvingPruner.cs
│   └── Models/
│       ├── Trial.cs
│       ├── Study.cs
│       └── TrialState.cs
│
└── ModelRegistry/
    ├── IModelRegistry.cs             # Registry interface
    ├── FileSystemRegistry.cs         # File-based implementation
    ├── Models/
    │   ├── RegisteredModel.cs
    │   ├── ModelVersion.cs
    │   └── ModelStage.cs
    └── Serialization/
        ├── IModelSerializer.cs       # Model serialization interface
        └── BinaryModelSerializer.cs  # Binary serialization
```

---

## Phase 1: Core Training Infrastructure

### Step 1: Define Training Configuration

**File**: `src/Training/Core/TrainingConfig.cs`

```csharp
namespace AiDotNet.Training.Core;

/// <summary>
/// Configuration for model training.
/// </summary>
public class TrainingConfig
{
    /// <summary>
    /// Number of training epochs.
    /// </summary>
    public int Epochs { get; set; } = 10;

    /// <summary>
    /// Batch size for training.
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Whether to perform validation after each epoch.
    /// </summary>
    public bool ValidateEveryEpoch { get; set; } = true;

    /// <summary>
    /// Whether to save checkpoints during training.
    /// </summary>
    public bool SaveCheckpoints { get; set; } = true;

    /// <summary>
    /// Directory to save checkpoints.
    /// </summary>
    public string CheckpointDirectory { get; set; } = "./checkpoints";

    /// <summary>
    /// Save checkpoint every N epochs (0 = only save best).
    /// </summary>
    public int CheckpointEveryNEpochs { get; set; } = 0;

    /// <summary>
    /// Metric to monitor for checkpointing (e.g., "val_accuracy", "val_loss").
    /// </summary>
    public string MonitorMetric { get; set; } = "val_loss";

    /// <summary>
    /// Whether higher metric values are better (true for accuracy, false for loss).
    /// </summary>
    public bool HigherIsBetter { get; set; } = false;

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Whether to display progress bar during training.
    /// </summary>
    public bool ShowProgress { get; set; } = true;
}
```

### Step 2: Define Metric Interface

**File**: `src/Training/Core/Metrics/IMetric.cs`

```csharp
namespace AiDotNet.Training.Core.Metrics;

/// <summary>
/// Interface for training and validation metrics.
/// </summary>
public interface IMetric<TInput, TOutput>
{
    /// <summary>
    /// Name of the metric (e.g., "accuracy", "loss").
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Compute metric for a batch of predictions and targets.
    /// </summary>
    double Compute(TOutput predictions, TOutput targets);

    /// <summary>
    /// Reset accumulated metric state.
    /// </summary>
    void Reset();

    /// <summary>
    /// Update metric with a batch result.
    /// </summary>
    void Update(TOutput predictions, TOutput targets);

    /// <summary>
    /// Get the aggregated metric value.
    /// </summary>
    double GetValue();
}
```

### Step 3: Implement Accuracy Metric

**File**: `src/Training/Core/Metrics/Accuracy.cs`

```csharp
namespace AiDotNet.Training.Core.Metrics;

using AiDotNet.LinearAlgebra;

/// <summary>
/// Classification accuracy metric.
/// </summary>
public class Accuracy : IMetric<Vector<double>, Vector<double>>
{
    private int _correct;
    private int _total;

    public string Name => "accuracy";

    public double Compute(Vector<double> predictions, Vector<double> targets)
    {
        if (predictions.Length != targets.Length)
            throw new ArgumentException("Predictions and targets must have same length");

        int correct = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            // For multi-class: argmax of predictions should match target
            if (Math.Abs(predictions[i] - targets[i]) < 1e-6)
                correct++;
        }

        return (double)correct / predictions.Length;
    }

    public void Reset()
    {
        _correct = 0;
        _total = 0;
    }

    public void Update(Vector<double> predictions, Vector<double> targets)
    {
        for (int i = 0; i < predictions.Length; i++)
        {
            if (Math.Abs(predictions[i] - targets[i]) < 1e-6)
                _correct++;
        }
        _total += predictions.Length;
    }

    public double GetValue()
    {
        return _total > 0 ? (double)_correct / _total : 0.0;
    }
}
```

### Step 4: Define Callback Interface

**File**: `src/Training/Core/Callbacks/ICallback.cs`

```csharp
namespace AiDotNet.Training.Core.Callbacks;

/// <summary>
/// Interface for training callbacks.
/// </summary>
public interface ICallback
{
    /// <summary>
    /// Called before training starts.
    /// </summary>
    void OnTrainingStart(TrainingContext context);

    /// <summary>
    /// Called after training completes.
    /// </summary>
    void OnTrainingEnd(TrainingContext context);

    /// <summary>
    /// Called at the start of each epoch.
    /// </summary>
    void OnEpochStart(int epoch, TrainingContext context);

    /// <summary>
    /// Called at the end of each epoch.
    /// </summary>
    void OnEpochEnd(int epoch, Dictionary<string, double> metrics, TrainingContext context);

    /// <summary>
    /// Called after each training batch.
    /// </summary>
    void OnBatchEnd(int batch, Dictionary<string, double> metrics, TrainingContext context);
}

/// <summary>
/// Context passed to callbacks containing training state.
/// </summary>
public class TrainingContext
{
    public IModel Model { get; set; }
    public IOptimizer Optimizer { get; set; }
    public TrainingConfig Config { get; set; }
    public Dictionary<string, object> State { get; set; } = new();
}
```

### Step 5: Implement Early Stopping Callback

**File**: `src/Training/Core/Callbacks/EarlyStopping.cs`

```csharp
namespace AiDotNet.Training.Core.Callbacks;

/// <summary>
/// Stop training when monitored metric stops improving.
/// </summary>
public class EarlyStopping : ICallback
{
    private readonly string _monitorMetric;
    private readonly int _patience;
    private readonly bool _higherIsBetter;
    private double _bestValue;
    private int _waitCount;

    public EarlyStopping(string monitorMetric = "val_loss", int patience = 5, bool higherIsBetter = false)
    {
        _monitorMetric = monitorMetric;
        _patience = patience;
        _higherIsBetter = higherIsBetter;
        _bestValue = higherIsBetter ? double.MinValue : double.MaxValue;
        _waitCount = 0;
    }

    public void OnTrainingStart(TrainingContext context) { }

    public void OnTrainingEnd(TrainingContext context) { }

    public void OnEpochStart(int epoch, TrainingContext context) { }

    public void OnEpochEnd(int epoch, Dictionary<string, double> metrics, TrainingContext context)
    {
        if (!metrics.ContainsKey(_monitorMetric))
            return;

        double currentValue = metrics[_monitorMetric];
        bool improved = _higherIsBetter
            ? currentValue > _bestValue
            : currentValue < _bestValue;

        if (improved)
        {
            _bestValue = currentValue;
            _waitCount = 0;
        }
        else
        {
            _waitCount++;
            if (_waitCount >= _patience)
            {
                Console.WriteLine($"Early stopping triggered after {epoch + 1} epochs");
                context.State["stop_training"] = true;
            }
        }
    }

    public void OnBatchEnd(int batch, Dictionary<string, double> metrics, TrainingContext context) { }
}
```

### Step 6: Implement Main Trainer Class

**File**: `src/Training/Core/Trainer.cs`

```csharp
namespace AiDotNet.Training.Core;

using AiDotNet.Training.Core.Callbacks;
using AiDotNet.Training.Core.Metrics;

/// <summary>
/// Main training orchestrator for machine learning models.
/// </summary>
public class Trainer<TInput, TOutput>
{
    private readonly IModel<TInput, TOutput> _model;
    private readonly IOptimizer _optimizer;
    private readonly ILoss<TOutput> _lossFunction;
    private readonly List<ICallback> _callbacks = new();
    private readonly List<IMetric<TInput, TOutput>> _metrics = new();

    public Trainer(
        IModel<TInput, TOutput> model,
        IOptimizer optimizer,
        ILoss<TOutput> lossFunction)
    {
        _model = model;
        _optimizer = optimizer;
        _lossFunction = lossFunction;
    }

    /// <summary>
    /// Add a callback to the trainer.
    /// </summary>
    public void AddCallback(ICallback callback)
    {
        _callbacks.Add(callback);
    }

    /// <summary>
    /// Add a metric to track during training.
    /// </summary>
    public void AddMetric(IMetric<TInput, TOutput> metric)
    {
        _metrics.Add(metric);
    }

    /// <summary>
    /// Train the model.
    /// </summary>
    public TrainingHistory Train(
        IDataLoader<TInput, TOutput> trainLoader,
        IDataLoader<TInput, TOutput> validationLoader = null,
        TrainingConfig config = null)
    {
        config = config ?? new TrainingConfig();
        var history = new TrainingHistory();
        var context = new TrainingContext
        {
            Model = _model,
            Optimizer = _optimizer,
            Config = config
        };

        // Invoke training start callbacks
        foreach (var callback in _callbacks)
            callback.OnTrainingStart(context);

        for (int epoch = 0; epoch < config.Epochs; epoch++)
        {
            // Check if training should stop
            if (context.State.ContainsKey("stop_training") && (bool)context.State["stop_training"])
                break;

            // Invoke epoch start callbacks
            foreach (var callback in _callbacks)
                callback.OnEpochStart(epoch, context);

            // Training phase
            var trainMetrics = TrainEpoch(trainLoader, epoch, context);
            history.AddEpochMetrics("train", trainMetrics);

            // Validation phase
            if (validationLoader != null && config.ValidateEveryEpoch)
            {
                var valMetrics = ValidateEpoch(validationLoader);
                history.AddEpochMetrics("val", valMetrics);
                trainMetrics = trainMetrics.Concat(valMetrics.Select(kv => new KeyValuePair<string, double>($"val_{kv.Key}", kv.Value)))
                    .ToDictionary(kv => kv.Key, kv => kv.Value);
            }

            // Invoke epoch end callbacks
            foreach (var callback in _callbacks)
                callback.OnEpochEnd(epoch, trainMetrics, context);

            // Checkpoint saving
            if (config.SaveCheckpoints)
            {
                SaveCheckpoint(epoch, trainMetrics, config);
            }
        }

        // Invoke training end callbacks
        foreach (var callback in _callbacks)
            callback.OnTrainingEnd(context);

        return history;
    }

    private Dictionary<string, double> TrainEpoch(IDataLoader<TInput, TOutput> loader, int epoch, TrainingContext context)
    {
        _model.Train();  // Set model to training mode

        // Reset metrics
        foreach (var metric in _metrics)
            metric.Reset();

        double totalLoss = 0.0;
        int batchCount = 0;

        foreach (var batch in loader)
        {
            // Forward pass
            var predictions = _model.Forward(batch.Input);

            // Compute loss
            var loss = _lossFunction.Compute(predictions, batch.Target);
            totalLoss += loss;

            // Backward pass
            var gradients = _lossFunction.Backward();

            // Update weights
            _optimizer.Step(gradients);

            // Update metrics
            foreach (var metric in _metrics)
                metric.Update(predictions, batch.Target);

            batchCount++;

            // Invoke batch end callbacks
            var batchMetrics = new Dictionary<string, double> { { "loss", loss } };
            foreach (var metric in _metrics)
                batchMetrics[metric.Name] = metric.GetValue();

            foreach (var callback in _callbacks)
                callback.OnBatchEnd(batchCount, batchMetrics, context);
        }

        // Aggregate metrics
        var metrics = new Dictionary<string, double>
        {
            { "loss", totalLoss / batchCount }
        };

        foreach (var metric in _metrics)
            metrics[metric.Name] = metric.GetValue();

        return metrics;
    }

    private Dictionary<string, double> ValidateEpoch(IDataLoader<TInput, TOutput> loader)
    {
        _model.Eval();  // Set model to evaluation mode

        // Reset metrics
        foreach (var metric in _metrics)
            metric.Reset();

        double totalLoss = 0.0;
        int batchCount = 0;

        foreach (var batch in loader)
        {
            // Forward pass (no gradients)
            var predictions = _model.Forward(batch.Input);

            // Compute loss
            var loss = _lossFunction.Compute(predictions, batch.Target);
            totalLoss += loss;

            // Update metrics
            foreach (var metric in _metrics)
                metric.Update(predictions, batch.Target);

            batchCount++;
        }

        // Aggregate metrics
        var metrics = new Dictionary<string, double>
        {
            { "loss", totalLoss / batchCount }
        };

        foreach (var metric in _metrics)
            metrics[metric.Name] = metric.GetValue();

        return metrics;
    }

    private void SaveCheckpoint(int epoch, Dictionary<string, double> metrics, TrainingConfig config)
    {
        // TODO: Implement checkpoint saving logic
        // This will save model state to disk
    }
}

/// <summary>
/// Stores training history (metrics over epochs).
/// </summary>
public class TrainingHistory
{
    private readonly Dictionary<string, List<double>> _history = new();

    public void AddEpochMetrics(string prefix, Dictionary<string, double> metrics)
    {
        foreach (var (key, value) in metrics)
        {
            string fullKey = $"{prefix}_{key}";
            if (!_history.ContainsKey(fullKey))
                _history[fullKey] = new List<double>();

            _history[fullKey].Add(value);
        }
    }

    public List<double> GetMetric(string key)
    {
        return _history.ContainsKey(key) ? _history[key] : new List<double>();
    }

    public Dictionary<string, List<double>> GetAllMetrics()
    {
        return new Dictionary<string, List<double>>(_history);
    }
}
```

### Testing Phase 1

**File**: `tests/UnitTests/Training/TrainerTests.cs`

```csharp
namespace AiDotNet.Tests.Training;

using Xunit;
using AiDotNet.Training.Core;

public class TrainerTests
{
    [Fact]
    public void Train_SimpleModel_ReturnsHistory()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new SGD(learningRate: 0.01);
        var loss = new MeanSquaredError();
        var trainer = new Trainer<Vector<double>, Vector<double>>(model, optimizer, loss);

        var trainLoader = CreateMockDataLoader(samples: 100, batchSize: 10);
        var config = new TrainingConfig { Epochs = 5 };

        // Act
        var history = trainer.Train(trainLoader, config: config);

        // Assert
        var trainLoss = history.GetMetric("train_loss");
        Assert.Equal(5, trainLoss.Count);  // 5 epochs
        Assert.True(trainLoss[0] > trainLoss[4]);  // Loss should decrease
    }

    [Fact]
    public void Train_WithEarlyStopping_StopsEarly()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new SGD(learningRate: 0.01);
        var loss = new MeanSquaredError();
        var trainer = new Trainer<Vector<double>, Vector<double>>(model, optimizer, loss);

        trainer.AddCallback(new EarlyStopping(patience: 2));

        var trainLoader = CreateMockDataLoader(samples: 100, batchSize: 10);
        var valLoader = CreateMockDataLoader(samples: 20, batchSize: 10);
        var config = new TrainingConfig { Epochs = 100, ValidateEveryEpoch = true };

        // Act
        var history = trainer.Train(trainLoader, valLoader, config);

        // Assert
        var trainLoss = history.GetMetric("train_loss");
        Assert.True(trainLoss.Count < 100);  // Should stop before 100 epochs
    }
}
```

---

## Phase 2: Experiment Tracking

### Step 1: Define Core Models

**File**: `src/Training/ExperimentTracking/Models/Experiment.cs`

```csharp
namespace AiDotNet.Training.ExperimentTracking.Models;

/// <summary>
/// Represents an ML experiment (collection of related runs).
/// </summary>
public class Experiment
{
    public string ExperimentId { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public Dictionary<string, string> Tags { get; set; } = new();
    public List<string> RunIds { get; set; } = new();
}
```

**File**: `src/Training/ExperimentTracking/Models/Run.cs`

```csharp
namespace AiDotNet.Training.ExperimentTracking.Models;

/// <summary>
/// Represents a single training run within an experiment.
/// </summary>
public class Run
{
    public string RunId { get; set; } = Guid.NewGuid().ToString();
    public string ExperimentId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; } = DateTime.UtcNow;
    public DateTime? EndTime { get; set; }
    public RunStatus Status { get; set; } = RunStatus.Running;
    public Dictionary<string, object> Parameters { get; set; } = new();
    public Dictionary<string, List<MetricEntry>> Metrics { get; set; } = new();
    public Dictionary<string, string> Tags { get; set; } = new();
    public List<string> ArtifactPaths { get; set; } = new();
}

public enum RunStatus
{
    Running,
    Finished,
    Failed,
    Killed
}
```

**File**: `src/Training/ExperimentTracking/Models/MetricEntry.cs`

```csharp
namespace AiDotNet.Training.ExperimentTracking.Models;

/// <summary>
/// A single metric value at a specific step/timestamp.
/// </summary>
public class MetricEntry
{
    public double Value { get; set; }
    public long Step { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}
```

### Step 2: Define Tracker Interface

**File**: `src/Training/ExperimentTracking/IExperimentTracker.cs`

```csharp
namespace AiDotNet.Training.ExperimentTracking;

using AiDotNet.Training.ExperimentTracking.Models;

/// <summary>
/// Interface for tracking ML experiments (MLflow-like).
/// </summary>
public interface IExperimentTracker
{
    // Experiment management
    Experiment CreateExperiment(string name, string description = null);
    Experiment GetExperiment(string experimentId);
    List<Experiment> ListExperiments();
    void DeleteExperiment(string experimentId);

    // Run management
    Run StartRun(string experimentId, string runName = null);
    void EndRun(string runId, RunStatus status = RunStatus.Finished);
    Run GetRun(string runId);
    List<Run> ListRuns(string experimentId);

    // Parameter logging
    void LogParameter(string runId, string key, object value);
    void LogParameters(string runId, Dictionary<string, object> parameters);

    // Metric logging
    void LogMetric(string runId, string key, double value, long step = 0);
    void LogMetrics(string runId, Dictionary<string, double> metrics, long step = 0);

    // Tag management
    void SetTag(string runId, string key, string value);
    void SetTags(string runId, Dictionary<string, string> tags);

    // Artifact logging
    void LogArtifact(string runId, string artifactName, byte[] content);
    void LogArtifact(string runId, string artifactName, string localPath);
    byte[] LoadArtifact(string runId, string artifactName);
}
```

### Step 3: Implement File System Tracker

**File**: `src/Training/ExperimentTracking/FileSystemTracker.cs`

```csharp
namespace AiDotNet.Training.ExperimentTracking;

using System.Text.Json;
using AiDotNet.Training.ExperimentTracking.Models;

/// <summary>
/// File system-based implementation of experiment tracker.
/// </summary>
public class FileSystemTracker : IExperimentTracker
{
    private readonly string _basePath;
    private readonly JsonSerializerOptions _jsonOptions;

    public FileSystemTracker(string basePath = "./mlruns")
    {
        _basePath = basePath;
        _jsonOptions = new JsonSerializerOptions { WriteIndented = true };

        Directory.CreateDirectory(_basePath);
    }

    public Experiment CreateExperiment(string name, string description = null)
    {
        var experiment = new Experiment
        {
            Name = name,
            Description = description ?? string.Empty
        };

        string experimentPath = GetExperimentPath(experiment.ExperimentId);
        Directory.CreateDirectory(experimentPath);

        SaveExperiment(experiment);
        return experiment;
    }

    public Experiment GetExperiment(string experimentId)
    {
        string metaPath = Path.Combine(GetExperimentPath(experimentId), "meta.json");
        if (!File.Exists(metaPath))
            throw new FileNotFoundException($"Experiment {experimentId} not found");

        string json = File.ReadAllText(metaPath);
        return JsonSerializer.Deserialize<Experiment>(json, _jsonOptions);
    }

    public List<Experiment> ListExperiments()
    {
        var experiments = new List<Experiment>();

        foreach (var dir in Directory.GetDirectories(_basePath))
        {
            string metaPath = Path.Combine(dir, "meta.json");
            if (File.Exists(metaPath))
            {
                string json = File.ReadAllText(metaPath);
                experiments.Add(JsonSerializer.Deserialize<Experiment>(json, _jsonOptions));
            }
        }

        return experiments;
    }

    public void DeleteExperiment(string experimentId)
    {
        string experimentPath = GetExperimentPath(experimentId);
        if (Directory.Exists(experimentPath))
            Directory.Delete(experimentPath, recursive: true);
    }

    public Run StartRun(string experimentId, string runName = null)
    {
        var run = new Run
        {
            ExperimentId = experimentId,
            StartTime = DateTime.UtcNow,
            Status = RunStatus.Running
        };

        if (!string.IsNullOrEmpty(runName))
            run.Tags["name"] = runName;

        string runPath = GetRunPath(experimentId, run.RunId);
        Directory.CreateDirectory(runPath);
        Directory.CreateDirectory(Path.Combine(runPath, "artifacts"));

        SaveRun(run);

        // Add run to experiment's run list
        var experiment = GetExperiment(experimentId);
        experiment.RunIds.Add(run.RunId);
        SaveExperiment(experiment);

        return run;
    }

    public void EndRun(string runId, RunStatus status = RunStatus.Finished)
    {
        var run = GetRunFromAnyExperiment(runId);
        run.EndTime = DateTime.UtcNow;
        run.Status = status;
        SaveRun(run);
    }

    public Run GetRun(string runId)
    {
        return GetRunFromAnyExperiment(runId);
    }

    public List<Run> ListRuns(string experimentId)
    {
        var runs = new List<Run>();
        string experimentPath = GetExperimentPath(experimentId);

        foreach (var dir in Directory.GetDirectories(experimentPath))
        {
            string runMetaPath = Path.Combine(dir, "run.json");
            if (File.Exists(runMetaPath))
            {
                string json = File.ReadAllText(runMetaPath);
                runs.Add(JsonSerializer.Deserialize<Run>(json, _jsonOptions));
            }
        }

        return runs;
    }

    public void LogParameter(string runId, string key, object value)
    {
        var run = GetRunFromAnyExperiment(runId);
        run.Parameters[key] = value;
        SaveRun(run);
    }

    public void LogParameters(string runId, Dictionary<string, object> parameters)
    {
        var run = GetRunFromAnyExperiment(runId);
        foreach (var (key, value) in parameters)
            run.Parameters[key] = value;
        SaveRun(run);
    }

    public void LogMetric(string runId, string key, double value, long step = 0)
    {
        var run = GetRunFromAnyExperiment(runId);

        if (!run.Metrics.ContainsKey(key))
            run.Metrics[key] = new List<MetricEntry>();

        run.Metrics[key].Add(new MetricEntry
        {
            Value = value,
            Step = step,
            Timestamp = DateTime.UtcNow
        });

        SaveRun(run);
    }

    public void LogMetrics(string runId, Dictionary<string, double> metrics, long step = 0)
    {
        foreach (var (key, value) in metrics)
            LogMetric(runId, key, value, step);
    }

    public void SetTag(string runId, string key, string value)
    {
        var run = GetRunFromAnyExperiment(runId);
        run.Tags[key] = value;
        SaveRun(run);
    }

    public void SetTags(string runId, Dictionary<string, string> tags)
    {
        var run = GetRunFromAnyExperiment(runId);
        foreach (var (key, value) in tags)
            run.Tags[key] = value;
        SaveRun(run);
    }

    public void LogArtifact(string runId, string artifactName, byte[] content)
    {
        var run = GetRunFromAnyExperiment(runId);
        string artifactPath = Path.Combine(GetRunPath(run.ExperimentId, runId), "artifacts", artifactName);

        Directory.CreateDirectory(Path.GetDirectoryName(artifactPath));
        File.WriteAllBytes(artifactPath, content);

        if (!run.ArtifactPaths.Contains(artifactName))
        {
            run.ArtifactPaths.Add(artifactName);
            SaveRun(run);
        }
    }

    public void LogArtifact(string runId, string artifactName, string localPath)
    {
        byte[] content = File.ReadAllBytes(localPath);
        LogArtifact(runId, artifactName, content);
    }

    public byte[] LoadArtifact(string runId, string artifactName)
    {
        var run = GetRunFromAnyExperiment(runId);
        string artifactPath = Path.Combine(GetRunPath(run.ExperimentId, runId), "artifacts", artifactName);

        if (!File.Exists(artifactPath))
            throw new FileNotFoundException($"Artifact {artifactName} not found");

        return File.ReadAllBytes(artifactPath);
    }

    // Helper methods

    private string GetExperimentPath(string experimentId)
    {
        return Path.Combine(_basePath, experimentId);
    }

    private string GetRunPath(string experimentId, string runId)
    {
        return Path.Combine(GetExperimentPath(experimentId), runId);
    }

    private void SaveExperiment(Experiment experiment)
    {
        string metaPath = Path.Combine(GetExperimentPath(experiment.ExperimentId), "meta.json");
        string json = JsonSerializer.Serialize(experiment, _jsonOptions);
        File.WriteAllText(metaPath, json);
    }

    private void SaveRun(Run run)
    {
        string runPath = Path.Combine(GetRunPath(run.ExperimentId, run.RunId), "run.json");
        string json = JsonSerializer.Serialize(run, _jsonOptions);
        File.WriteAllText(runPath, json);
    }

    private Run GetRunFromAnyExperiment(string runId)
    {
        foreach (var experiment in ListExperiments())
        {
            string runPath = Path.Combine(GetRunPath(experiment.ExperimentId, runId), "run.json");
            if (File.Exists(runPath))
            {
                string json = File.ReadAllText(runPath);
                return JsonSerializer.Deserialize<Run>(json, _jsonOptions);
            }
        }

        throw new FileNotFoundException($"Run {runId} not found");
    }
}
```

### Step 4: Integrate Tracker with Trainer

**File**: `src/Training/Core/Callbacks/ExperimentTrackerCallback.cs`

```csharp
namespace AiDotNet.Training.Core.Callbacks;

using AiDotNet.Training.ExperimentTracking;

/// <summary>
/// Callback that logs training to an experiment tracker.
/// </summary>
public class ExperimentTrackerCallback : ICallback
{
    private readonly IExperimentTracker _tracker;
    private readonly string _runId;

    public ExperimentTrackerCallback(IExperimentTracker tracker, string runId)
    {
        _tracker = tracker;
        _runId = runId;
    }

    public void OnTrainingStart(TrainingContext context)
    {
        // Log training configuration as parameters
        _tracker.LogParameter(_runId, "epochs", context.Config.Epochs);
        _tracker.LogParameter(_runId, "batch_size", context.Config.BatchSize);

        if (context.Config.Seed.HasValue)
            _tracker.LogParameter(_runId, "seed", context.Config.Seed.Value);
    }

    public void OnTrainingEnd(TrainingContext context)
    {
        _tracker.EndRun(_runId, ExperimentTracking.Models.RunStatus.Finished);
    }

    public void OnEpochStart(int epoch, TrainingContext context) { }

    public void OnEpochEnd(int epoch, Dictionary<string, double> metrics, TrainingContext context)
    {
        // Log all metrics for this epoch
        _tracker.LogMetrics(_runId, metrics, step: epoch);
    }

    public void OnBatchEnd(int batch, Dictionary<string, double> metrics, TrainingContext context) { }
}
```

### Testing Phase 2

**File**: `tests/UnitTests/Training/ExperimentTrackingTests.cs`

```csharp
namespace AiDotNet.Tests.Training;

using Xunit;
using AiDotNet.Training.ExperimentTracking;

public class ExperimentTrackingTests
{
    [Fact]
    public void CreateExperiment_ValidName_CreatesExperiment()
    {
        // Arrange
        var tracker = new FileSystemTracker("./test_mlruns");

        // Act
        var experiment = tracker.CreateExperiment("test_experiment");

        // Assert
        Assert.NotNull(experiment);
        Assert.Equal("test_experiment", experiment.Name);
        Assert.NotEmpty(experiment.ExperimentId);

        // Cleanup
        tracker.DeleteExperiment(experiment.ExperimentId);
    }

    [Fact]
    public void StartRun_ValidExperiment_CreatesRun()
    {
        // Arrange
        var tracker = new FileSystemTracker("./test_mlruns");
        var experiment = tracker.CreateExperiment("test_experiment");

        // Act
        var run = tracker.StartRun(experiment.ExperimentId, "test_run");

        // Assert
        Assert.NotNull(run);
        Assert.Equal(experiment.ExperimentId, run.ExperimentId);
        Assert.Equal(RunStatus.Running, run.Status);

        // Cleanup
        tracker.DeleteExperiment(experiment.ExperimentId);
    }

    [Fact]
    public void LogMetric_ValidRun_StoresMetric()
    {
        // Arrange
        var tracker = new FileSystemTracker("./test_mlruns");
        var experiment = tracker.CreateExperiment("test_experiment");
        var run = tracker.StartRun(experiment.ExperimentId);

        // Act
        tracker.LogMetric(run.RunId, "loss", 0.5, step: 0);
        tracker.LogMetric(run.RunId, "loss", 0.3, step: 1);

        // Assert
        var retrievedRun = tracker.GetRun(run.RunId);
        Assert.True(retrievedRun.Metrics.ContainsKey("loss"));
        Assert.Equal(2, retrievedRun.Metrics["loss"].Count);
        Assert.Equal(0.5, retrievedRun.Metrics["loss"][0].Value);
        Assert.Equal(0.3, retrievedRun.Metrics["loss"][1].Value);

        // Cleanup
        tracker.DeleteExperiment(experiment.ExperimentId);
    }
}
```

---

## Phase 3: Hyperparameter Optimization

### Step 1: Define Search Space

**File**: `src/Training/HyperparameterOptimization/SearchSpace/SearchSpace.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization.SearchSpace;

/// <summary>
/// Base class for hyperparameter search spaces.
/// </summary>
public abstract class SearchSpaceBase
{
    public abstract object Sample(Random rng);
    public abstract string GetValueType();
}
```

**File**: `src/Training/HyperparameterOptimization/SearchSpace/ContinuousSpace.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization.SearchSpace;

/// <summary>
/// Continuous (float) hyperparameter space.
/// </summary>
public class ContinuousSpace : SearchSpaceBase
{
    public double Low { get; set; }
    public double High { get; set; }
    public bool LogScale { get; set; }

    public ContinuousSpace(double low, double high, bool logScale = false)
    {
        if (logScale && low <= 0)
            throw new ArgumentException("Low must be > 0 for log scale");

        Low = logScale ? Math.Log(low) : low;
        High = logScale ? Math.Log(high) : high;
        LogScale = logScale;
    }

    public override object Sample(Random rng)
    {
        double value = rng.NextDouble() * (High - Low) + Low;
        return LogScale ? Math.Exp(value) : value;
    }

    public override string GetValueType() => "continuous";
}
```

**File**: `src/Training/HyperparameterOptimization/SearchSpace/DiscreteSpace.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization.SearchSpace;

/// <summary>
/// Discrete (categorical) hyperparameter space.
/// </summary>
public class DiscreteSpace : SearchSpaceBase
{
    public object[] Choices { get; set; }

    public DiscreteSpace(params object[] choices)
    {
        if (choices.Length == 0)
            throw new ArgumentException("Must provide at least one choice");

        Choices = choices;
    }

    public override object Sample(Random rng)
    {
        return Choices[rng.Next(Choices.Length)];
    }

    public override string GetValueType() => "categorical";
}
```

**File**: `src/Training/HyperparameterOptimization/SearchSpace/IntegerSpace.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization.SearchSpace;

/// <summary>
/// Integer hyperparameter space.
/// </summary>
public class IntegerSpace : SearchSpaceBase
{
    public int Low { get; set; }
    public int High { get; set; }

    public IntegerSpace(int low, int high)
    {
        if (low >= high)
            throw new ArgumentException("Low must be < High");

        Low = low;
        High = high;
    }

    public override object Sample(Random rng)
    {
        return rng.Next(Low, High + 1);
    }

    public override string GetValueType() => "int";
}
```

### Step 2: Define Trial and Study Models

**File**: `src/Training/HyperparameterOptimization/Models/Trial.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization.Models;

/// <summary>
/// Represents a single trial in hyperparameter optimization.
/// </summary>
public class Trial
{
    public int TrialId { get; set; }
    public Dictionary<string, object> Parameters { get; set; } = new();
    public double? ObjectiveValue { get; set; }
    public TrialState State { get; set; } = TrialState.Running;
    public DateTime StartTime { get; set; } = DateTime.UtcNow;
    public DateTime? EndTime { get; set; }
    public Dictionary<int, double> IntermediateValues { get; set; } = new();
    public Dictionary<string, object> UserAttributes { get; set; } = new();
}

public enum TrialState
{
    Running,
    Complete,
    Pruned,
    Failed
}
```

**File**: `src/Training/HyperparameterOptimization/Models/Study.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization.Models;

/// <summary>
/// Represents an optimization study (collection of trials).
/// </summary>
public class Study
{
    public string StudyId { get; set; } = Guid.NewGuid().ToString();
    public string StudyName { get; set; } = string.Empty;
    public OptimizationDirection Direction { get; set; } = OptimizationDirection.Maximize;
    public List<Trial> Trials { get; set; } = new();
    public Dictionary<string, SearchSpaceBase> SearchSpace { get; set; } = new();

    public Trial BestTrial =>
        Direction == OptimizationDirection.Maximize
            ? Trials.Where(t => t.State == TrialState.Complete)
                    .OrderByDescending(t => t.ObjectiveValue)
                    .FirstOrDefault()
            : Trials.Where(t => t.State == TrialState.Complete)
                    .OrderBy(t => t.ObjectiveValue)
                    .FirstOrDefault();
}

public enum OptimizationDirection
{
    Minimize,
    Maximize
}
```

### Step 3: Implement Random Search Optimizer

**File**: `src/Training/HyperparameterOptimization/RandomSearchOptimizer.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization;

using AiDotNet.Training.HyperparameterOptimization.Models;
using AiDotNet.Training.HyperparameterOptimization.SearchSpace;

/// <summary>
/// Random search hyperparameter optimizer.
/// </summary>
public class RandomSearchOptimizer : IHyperparameterOptimizer
{
    private readonly Random _rng;

    public RandomSearchOptimizer(int? seed = null)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public Dictionary<string, object> SuggestParameters(Study study)
    {
        var parameters = new Dictionary<string, object>();

        foreach (var (name, space) in study.SearchSpace)
        {
            parameters[name] = space.Sample(_rng);
        }

        return parameters;
    }

    public Study Optimize(
        Dictionary<string, SearchSpaceBase> searchSpace,
        Func<Dictionary<string, object>, double> objective,
        int nTrials,
        OptimizationDirection direction = OptimizationDirection.Maximize)
    {
        var study = new Study
        {
            SearchSpace = searchSpace,
            Direction = direction
        };

        for (int i = 0; i < nTrials; i++)
        {
            var trial = new Trial { TrialId = i };

            try
            {
                // Sample parameters
                trial.Parameters = SuggestParameters(study);

                // Evaluate objective
                trial.ObjectiveValue = objective(trial.Parameters);
                trial.State = TrialState.Complete;
                trial.EndTime = DateTime.UtcNow;

                Console.WriteLine($"Trial {i}: {trial.ObjectiveValue} (params: {string.Join(", ", trial.Parameters.Select(kv => $"{kv.Key}={kv.Value}"))})");
            }
            catch (Exception ex)
            {
                trial.State = TrialState.Failed;
                trial.EndTime = DateTime.UtcNow;
                trial.UserAttributes["error"] = ex.Message;
                Console.WriteLine($"Trial {i} failed: {ex.Message}");
            }

            study.Trials.Add(trial);
        }

        return study;
    }
}
```

### Step 4: Implement TPE Optimizer (Simplified)

**File**: `src/Training/HyperparameterOptimization/TPEOptimizer.cs`

```csharp
namespace AiDotNet.Training.HyperparameterOptimization;

using AiDotNet.Training.HyperparameterOptimization.Models;
using AiDotNet.Training.HyperparameterOptimization.SearchSpace;

/// <summary>
/// Tree-structured Parzen Estimator (TPE) optimizer.
/// Simplified implementation focusing on continuous parameters.
/// </summary>
public class TPEOptimizer : IHyperparameterOptimizer
{
    private readonly Random _rng;
    private readonly double _gamma;  // Quantile for splitting good/bad trials

    public TPEOptimizer(int? seed = null, double gamma = 0.25)
    {
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
        _gamma = gamma;
    }

    public Dictionary<string, object> SuggestParameters(Study study)
    {
        // If not enough trials, use random sampling
        if (study.Trials.Count < 10)
            return RandomSample(study.SearchSpace);

        var completedTrials = study.Trials
            .Where(t => t.State == TrialState.Complete)
            .OrderByDescending(t => study.Direction == OptimizationDirection.Maximize ? t.ObjectiveValue : -t.ObjectiveValue)
            .ToList();

        if (completedTrials.Count < 5)
            return RandomSample(study.SearchSpace);

        // Split trials into good (top gamma%) and bad (bottom 1-gamma%)
        int splitIndex = (int)(completedTrials.Count * _gamma);
        var goodTrials = completedTrials.Take(splitIndex).ToList();
        var badTrials = completedTrials.Skip(splitIndex).ToList();

        var parameters = new Dictionary<string, object>();

        foreach (var (name, space) in study.SearchSpace)
        {
            if (space is ContinuousSpace contSpace)
            {
                // Get values from good and bad trials
                var goodValues = goodTrials.Select(t => Convert.ToDouble(t.Parameters[name])).ToList();
                var badValues = badTrials.Select(t => Convert.ToDouble(t.Parameters[name])).ToList();

                // Fit Gaussian distributions
                var goodMean = goodValues.Average();
                var goodStd = Math.Sqrt(goodValues.Select(v => Math.Pow(v - goodMean, 2)).Average());

                var badMean = badValues.Average();
                var badStd = Math.Sqrt(badValues.Select(v => Math.Pow(v - badMean, 2)).Average());

                // Sample from good distribution (simplified TPE)
                double value = SampleGaussian(goodMean, goodStd);

                // Clamp to bounds
                double low = contSpace.LogScale ? Math.Exp(contSpace.Low) : contSpace.Low;
                double high = contSpace.LogScale ? Math.Exp(contSpace.High) : contSpace.High;
                value = Math.Clamp(value, low, high);

                parameters[name] = value;
            }
            else
            {
                // For non-continuous spaces, fall back to random sampling
                parameters[name] = space.Sample(_rng);
            }
        }

        return parameters;
    }

    public Study Optimize(
        Dictionary<string, SearchSpaceBase> searchSpace,
        Func<Dictionary<string, object>, double> objective,
        int nTrials,
        OptimizationDirection direction = OptimizationDirection.Maximize)
    {
        var study = new Study
        {
            SearchSpace = searchSpace,
            Direction = direction
        };

        for (int i = 0; i < nTrials; i++)
        {
            var trial = new Trial { TrialId = i };

            try
            {
                // Suggest parameters using TPE
                trial.Parameters = SuggestParameters(study);

                // Evaluate objective
                trial.ObjectiveValue = objective(trial.Parameters);
                trial.State = TrialState.Complete;
                trial.EndTime = DateTime.UtcNow;

                Console.WriteLine($"Trial {i}: {trial.ObjectiveValue} (params: {string.Join(", ", trial.Parameters.Select(kv => $"{kv.Key}={kv.Value:F4}"))})");
            }
            catch (Exception ex)
            {
                trial.State = TrialState.Failed;
                trial.EndTime = DateTime.UtcNow;
                trial.UserAttributes["error"] = ex.Message;
                Console.WriteLine($"Trial {i} failed: {ex.Message}");
            }

            study.Trials.Add(trial);
        }

        return study;
    }

    private Dictionary<string, object> RandomSample(Dictionary<string, SearchSpaceBase> searchSpace)
    {
        var parameters = new Dictionary<string, object>();
        foreach (var (name, space) in searchSpace)
            parameters[name] = space.Sample(_rng);
        return parameters;
    }

    private double SampleGaussian(double mean, double std)
    {
        // Box-Muller transform
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = 1.0 - _rng.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + std * randStdNormal;
    }
}

public interface IHyperparameterOptimizer
{
    Dictionary<string, object> SuggestParameters(Study study);
    Study Optimize(
        Dictionary<string, SearchSpaceBase> searchSpace,
        Func<Dictionary<string, object>, double> objective,
        int nTrials,
        OptimizationDirection direction = OptimizationDirection.Maximize);
}
```

### Testing Phase 3

**File**: `tests/UnitTests/Training/HyperparameterOptimizationTests.cs`

```csharp
namespace AiDotNet.Tests.Training;

using Xunit;
using AiDotNet.Training.HyperparameterOptimization;
using AiDotNet.Training.HyperparameterOptimization.SearchSpace;

public class HyperparameterOptimizationTests
{
    [Fact]
    public void RandomSearch_SimpleProblem_FindsOptimum()
    {
        // Arrange: Optimize f(x) = -(x - 5)^2 (maximum at x=5)
        var searchSpace = new Dictionary<string, SearchSpaceBase>
        {
            { "x", new ContinuousSpace(0, 10) }
        };

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = Convert.ToDouble(parameters["x"]);
            return -Math.Pow(x - 5, 2);  // Maximum at x=5
        };

        var optimizer = new RandomSearchOptimizer(seed: 42);

        // Act
        var study = optimizer.Optimize(searchSpace, objective, nTrials: 50, OptimizationDirection.Maximize);

        // Assert
        Assert.Equal(50, study.Trials.Count);
        Assert.NotNull(study.BestTrial);

        double bestX = Convert.ToDouble(study.BestTrial.Parameters["x"]);
        Assert.True(Math.Abs(bestX - 5) < 1.0);  // Should be close to 5
    }

    [Fact]
    public void TPEOptimizer_SimpleProblem_ConvergesFaster()
    {
        // Arrange
        var searchSpace = new Dictionary<string, SearchSpaceBase>
        {
            { "x", new ContinuousSpace(0, 10) }
        };

        Func<Dictionary<string, object>, double> objective = (parameters) =>
        {
            double x = Convert.ToDouble(parameters["x"]);
            return -Math.Pow(x - 5, 2);
        };

        var optimizer = new TPEOptimizer(seed: 42);

        // Act
        var study = optimizer.Optimize(searchSpace, objective, nTrials: 30, OptimizationDirection.Maximize);

        // Assert
        double bestX = Convert.ToDouble(study.BestTrial.Parameters["x"]);
        Assert.True(Math.Abs(bestX - 5) < 0.5);  // TPE should converge closer
    }
}
```

---

## Phase 4: Model Registry

### Step 1: Define Registry Models

**File**: `src/Training/ModelRegistry/Models/RegisteredModel.cs`

```csharp
namespace AiDotNet.Training.ModelRegistry.Models;

/// <summary>
/// Represents a registered model in the registry.
/// </summary>
public class RegisteredModel
{
    public string ModelId { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? LastUpdated { get; set; }
    public Dictionary<string, string> Tags { get; set; } = new();
    public List<string> VersionIds { get; set; } = new();
}
```

**File**: `src/Training/ModelRegistry/Models/ModelVersion.cs`

```csharp
namespace AiDotNet.Training.ModelRegistry.Models;

/// <summary>
/// Represents a specific version of a registered model.
/// </summary>
public class ModelVersion
{
    public string VersionId { get; set; } = Guid.NewGuid().ToString();
    public string ModelId { get; set; } = string.Empty;
    public int Version { get; set; }
    public string RunId { get; set; } = string.Empty;  // Link to training run
    public ModelStage Stage { get; set; } = ModelStage.Development;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public string StoragePath { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; set; } = new();
    public Dictionary<string, double> Metrics { get; set; } = new();
    public Dictionary<string, string> Tags { get; set; } = new();
}

public enum ModelStage
{
    Development,
    Staging,
    Production,
    Archived
}
```

### Step 2: Define Registry Interface

**File**: `src/Training/ModelRegistry/IModelRegistry.cs`

```csharp
namespace AiDotNet.Training.ModelRegistry;

using AiDotNet.Training.ModelRegistry.Models;

/// <summary>
/// Interface for model registry (MLflow Model Registry-like).
/// </summary>
public interface IModelRegistry
{
    // Model management
    RegisteredModel RegisterModel(string name, string description = null);
    RegisteredModel GetModel(string modelId);
    RegisteredModel GetModelByName(string name);
    List<RegisteredModel> ListModels();
    void DeleteModel(string modelId);

    // Version management
    ModelVersion CreateModelVersion(
        string modelId,
        byte[] modelData,
        string runId = null,
        Dictionary<string, object> metadata = null,
        Dictionary<string, double> metrics = null);

    ModelVersion GetModelVersion(string versionId);
    List<ModelVersion> ListModelVersions(string modelId);
    void DeleteModelVersion(string versionId);

    // Stage management
    void SetModelStage(string versionId, ModelStage stage);
    ModelVersion GetProductionModel(string modelName);
    ModelVersion GetStagingModel(string modelName);

    // Model loading
    byte[] LoadModelData(string versionId);

    // Tags
    void SetModelTag(string modelId, string key, string value);
    void SetVersionTag(string versionId, string key, string value);
}
```

### Step 3: Implement File System Registry

**File**: `src/Training/ModelRegistry/FileSystemRegistry.cs`

```csharp
namespace AiDotNet.Training.ModelRegistry;

using System.Text.Json;
using AiDotNet.Training.ModelRegistry.Models;

/// <summary>
/// File system-based model registry implementation.
/// </summary>
public class FileSystemRegistry : IModelRegistry
{
    private readonly string _basePath;
    private readonly JsonSerializerOptions _jsonOptions;

    public FileSystemRegistry(string basePath = "./model_registry")
    {
        _basePath = basePath;
        _jsonOptions = new JsonSerializerOptions { WriteIndented = true };

        Directory.CreateDirectory(_basePath);
    }

    public RegisteredModel RegisterModel(string name, string description = null)
    {
        // Check if model with this name already exists
        var existing = GetModelByName(name);
        if (existing != null)
            return existing;

        var model = new RegisteredModel
        {
            Name = name,
            Description = description ?? string.Empty
        };

        string modelPath = GetModelPath(model.ModelId);
        Directory.CreateDirectory(modelPath);

        SaveModel(model);
        return model;
    }

    public RegisteredModel GetModel(string modelId)
    {
        string metaPath = Path.Combine(GetModelPath(modelId), "meta.json");
        if (!File.Exists(metaPath))
            throw new FileNotFoundException($"Model {modelId} not found");

        string json = File.ReadAllText(metaPath);
        return JsonSerializer.Deserialize<RegisteredModel>(json, _jsonOptions);
    }

    public RegisteredModel GetModelByName(string name)
    {
        foreach (var model in ListModels())
        {
            if (model.Name == name)
                return model;
        }
        return null;
    }

    public List<RegisteredModel> ListModels()
    {
        var models = new List<RegisteredModel>();

        foreach (var dir in Directory.GetDirectories(_basePath))
        {
            string metaPath = Path.Combine(dir, "meta.json");
            if (File.Exists(metaPath))
            {
                string json = File.ReadAllText(metaPath);
                models.Add(JsonSerializer.Deserialize<RegisteredModel>(json, _jsonOptions));
            }
        }

        return models;
    }

    public void DeleteModel(string modelId)
    {
        string modelPath = GetModelPath(modelId);
        if (Directory.Exists(modelPath))
            Directory.Delete(modelPath, recursive: true);
    }

    public ModelVersion CreateModelVersion(
        string modelId,
        byte[] modelData,
        string runId = null,
        Dictionary<string, object> metadata = null,
        Dictionary<string, double> metrics = null)
    {
        var model = GetModel(modelId);

        int newVersion = model.VersionIds.Count + 1;

        var version = new ModelVersion
        {
            ModelId = modelId,
            Version = newVersion,
            RunId = runId ?? string.Empty,
            Metadata = metadata ?? new Dictionary<string, object>(),
            Metrics = metrics ?? new Dictionary<string, double>()
        };

        // Save model data
        string versionPath = GetVersionPath(modelId, version.VersionId);
        Directory.CreateDirectory(versionPath);

        string dataPath = Path.Combine(versionPath, "model.bin");
        File.WriteAllBytes(dataPath, modelData);
        version.StoragePath = dataPath;

        // Save version metadata
        SaveVersion(version);

        // Update model
        model.VersionIds.Add(version.VersionId);
        model.LastUpdated = DateTime.UtcNow;
        SaveModel(model);

        return version;
    }

    public ModelVersion GetModelVersion(string versionId)
    {
        foreach (var model in ListModels())
        {
            string versionPath = Path.Combine(GetModelPath(model.ModelId), versionId, "version.json");
            if (File.Exists(versionPath))
            {
                string json = File.ReadAllText(versionPath);
                return JsonSerializer.Deserialize<ModelVersion>(json, _jsonOptions);
            }
        }

        throw new FileNotFoundException($"Version {versionId} not found");
    }

    public List<ModelVersion> ListModelVersions(string modelId)
    {
        var versions = new List<ModelVersion>();
        string modelPath = GetModelPath(modelId);

        foreach (var dir in Directory.GetDirectories(modelPath))
        {
            string versionPath = Path.Combine(dir, "version.json");
            if (File.Exists(versionPath))
            {
                string json = File.ReadAllText(versionPath);
                versions.Add(JsonSerializer.Deserialize<ModelVersion>(json, _jsonOptions));
            }
        }

        return versions.OrderBy(v => v.Version).ToList();
    }

    public void DeleteModelVersion(string versionId)
    {
        var version = GetModelVersion(versionId);
        string versionPath = GetVersionPath(version.ModelId, versionId);

        if (Directory.Exists(versionPath))
            Directory.Delete(versionPath, recursive: true);

        // Update model's version list
        var model = GetModel(version.ModelId);
        model.VersionIds.Remove(versionId);
        SaveModel(model);
    }

    public void SetModelStage(string versionId, ModelStage stage)
    {
        var version = GetModelVersion(versionId);

        // If setting to production/staging, demote existing production/staging versions
        if (stage == ModelStage.Production || stage == ModelStage.Staging)
        {
            foreach (var v in ListModelVersions(version.ModelId))
            {
                if (v.VersionId != versionId && v.Stage == stage)
                {
                    v.Stage = ModelStage.Development;
                    SaveVersion(v);
                }
            }
        }

        version.Stage = stage;
        SaveVersion(version);
    }

    public ModelVersion GetProductionModel(string modelName)
    {
        var model = GetModelByName(modelName);
        if (model == null)
            return null;

        return ListModelVersions(model.ModelId)
            .FirstOrDefault(v => v.Stage == ModelStage.Production);
    }

    public ModelVersion GetStagingModel(string modelName)
    {
        var model = GetModelByName(modelName);
        if (model == null)
            return null;

        return ListModelVersions(model.ModelId)
            .FirstOrDefault(v => v.Stage == ModelStage.Staging);
    }

    public byte[] LoadModelData(string versionId)
    {
        var version = GetModelVersion(versionId);

        if (!File.Exists(version.StoragePath))
            throw new FileNotFoundException($"Model data not found at {version.StoragePath}");

        return File.ReadAllBytes(version.StoragePath);
    }

    public void SetModelTag(string modelId, string key, string value)
    {
        var model = GetModel(modelId);
        model.Tags[key] = value;
        SaveModel(model);
    }

    public void SetVersionTag(string versionId, string key, string value)
    {
        var version = GetModelVersion(versionId);
        version.Tags[key] = value;
        SaveVersion(version);
    }

    // Helper methods

    private string GetModelPath(string modelId)
    {
        return Path.Combine(_basePath, modelId);
    }

    private string GetVersionPath(string modelId, string versionId)
    {
        return Path.Combine(GetModelPath(modelId), versionId);
    }

    private void SaveModel(RegisteredModel model)
    {
        string metaPath = Path.Combine(GetModelPath(model.ModelId), "meta.json");
        string json = JsonSerializer.Serialize(model, _jsonOptions);
        File.WriteAllText(metaPath, json);
    }

    private void SaveVersion(ModelVersion version)
    {
        string versionPath = Path.Combine(GetVersionPath(version.ModelId, version.VersionId), "version.json");
        string json = JsonSerializer.Serialize(version, _jsonOptions);
        File.WriteAllText(versionPath, json);
    }
}
```

### Testing Phase 4

**File**: `tests/UnitTests/Training/ModelRegistryTests.cs`

```csharp
namespace AiDotNet.Tests.Training;

using Xunit;
using AiDotNet.Training.ModelRegistry;
using AiDotNet.Training.ModelRegistry.Models;

public class ModelRegistryTests
{
    [Fact]
    public void RegisterModel_ValidName_CreatesModel()
    {
        // Arrange
        var registry = new FileSystemRegistry("./test_registry");

        // Act
        var model = registry.RegisterModel("test_model", "Test model description");

        // Assert
        Assert.NotNull(model);
        Assert.Equal("test_model", model.Name);
        Assert.NotEmpty(model.ModelId);

        // Cleanup
        registry.DeleteModel(model.ModelId);
    }

    [Fact]
    public void CreateModelVersion_ValidModel_CreatesVersion()
    {
        // Arrange
        var registry = new FileSystemRegistry("./test_registry");
        var model = registry.RegisterModel("test_model");
        byte[] modelData = new byte[] { 1, 2, 3, 4, 5 };

        // Act
        var version = registry.CreateModelVersion(model.ModelId, modelData);

        // Assert
        Assert.NotNull(version);
        Assert.Equal(model.ModelId, version.ModelId);
        Assert.Equal(1, version.Version);
        Assert.Equal(ModelStage.Development, version.Stage);

        // Cleanup
        registry.DeleteModel(model.ModelId);
    }

    [Fact]
    public void SetModelStage_ToProduction_UpdatesStage()
    {
        // Arrange
        var registry = new FileSystemRegistry("./test_registry");
        var model = registry.RegisterModel("test_model");
        byte[] modelData = new byte[] { 1, 2, 3 };
        var version = registry.CreateModelVersion(model.ModelId, modelData);

        // Act
        registry.SetModelStage(version.VersionId, ModelStage.Production);

        // Assert
        var productionModel = registry.GetProductionModel("test_model");
        Assert.NotNull(productionModel);
        Assert.Equal(version.VersionId, productionModel.VersionId);

        // Cleanup
        registry.DeleteModel(model.ModelId);
    }
}
```

---

## Testing Strategy

### Unit Tests

1. **Trainer Tests**: Test training loop, callbacks, metric tracking
2. **Experiment Tracker Tests**: Test experiment/run creation, metric logging
3. **HPO Tests**: Test search space sampling, optimization algorithms
4. **Registry Tests**: Test model registration, versioning, stage management

### Integration Tests

**File**: `tests/IntegrationTests/Training/EndToEndTrainingTests.cs`

```csharp
namespace AiDotNet.Tests.Integration.Training;

using Xunit;
using AiDotNet.Training.Core;
using AiDotNet.Training.ExperimentTracking;
using AiDotNet.Training.ModelRegistry;

public class EndToEndTrainingTests
{
    [Fact]
    public void CompleteWorkflow_TrainTrackRegister_WorksEndToEnd()
    {
        // Arrange
        var tracker = new FileSystemTracker("./test_mlruns");
        var registry = new FileSystemRegistry("./test_registry");

        var experiment = tracker.CreateExperiment("integration_test");
        var run = tracker.StartRun(experiment.ExperimentId, "test_run");

        var model = new MockModel();
        var optimizer = new SGD(learningRate: 0.01);
        var loss = new MeanSquaredError();
        var trainer = new Trainer<Vector<double>, Vector<double>>(model, optimizer, loss);

        // Add experiment tracking callback
        trainer.AddCallback(new ExperimentTrackerCallback(tracker, run.RunId));

        var trainLoader = CreateMockDataLoader(samples: 100, batchSize: 10);
        var config = new TrainingConfig { Epochs = 5 };

        // Act: Train model
        var history = trainer.Train(trainLoader, config: config);

        // Assert: Check training completed
        Assert.Equal(5, history.GetMetric("train_loss").Count);

        // Act: Register trained model
        var registeredModel = registry.RegisterModel("integration_test_model");
        byte[] modelData = SerializeModel(model);
        var version = registry.CreateModelVersion(
            registeredModel.ModelId,
            modelData,
            runId: run.RunId,
            metrics: new Dictionary<string, double>
            {
                { "final_loss", history.GetMetric("train_loss").Last() }
            });

        // Assert: Check model registered
        Assert.NotNull(version);
        Assert.Equal(run.RunId, version.RunId);

        // Cleanup
        tracker.DeleteExperiment(experiment.ExperimentId);
        registry.DeleteModel(registeredModel.ModelId);
    }
}
```

---

## Common Pitfalls

### 1. Memory Leaks in Training Loop

**Problem**: Not disposing of intermediate tensors/matrices

```csharp
// Bad: Memory leak
for (int epoch = 0; epoch < epochs; epoch++)
{
    foreach (var batch in loader)
    {
        var predictions = model.Forward(batch.Input);  // Allocates memory
        var loss = lossFunction.Compute(predictions, batch.Target);
        // predictions never disposed!
    }
}

// Good: Explicit disposal
for (int epoch = 0; epoch < epochs; epoch++)
{
    foreach (var batch in loader)
    {
        using (var predictions = model.Forward(batch.Input))
        {
            var loss = lossFunction.Compute(predictions, batch.Target);
        }
    }
}
```

### 2. Thread Safety in Experiment Tracking

**Problem**: Multiple threads writing to same run simultaneously

**Solution**: Use locks or thread-safe collections:
```csharp
private readonly object _lock = new object();

public void LogMetric(string runId, string key, double value, long step)
{
    lock (_lock)
    {
        // Write to file
    }
}
```

### 3. Hyperparameter Type Mismatches

**Problem**: Search space returns double, but training code expects int

```csharp
// Bad: Type error
var batchSize = (int)parameters["batch_size"];  // May throw if value is double

// Good: Explicit conversion
var batchSize = Convert.ToInt32(parameters["batch_size"]);
```

### 4. Model Stage Conflicts

**Problem**: Multiple versions set to production simultaneously

**Solution**: Automatically demote existing production models (implemented in `SetModelStage`)

### 5. Large Artifact Storage

**Problem**: Storing huge model files in memory

**Solution**: Stream file writes:
```csharp
public void LogArtifact(string runId, string artifactName, Stream stream)
{
    string artifactPath = GetArtifactPath(runId, artifactName);
    using (var fileStream = File.Create(artifactPath))
    {
        stream.CopyTo(fileStream);
    }
}
```

---

## Summary

This guide covered:

1. **Training Infrastructure**: Trainer class, callbacks, metrics
2. **Experiment Tracking**: MLflow-like tracking with runs, experiments, metrics
3. **Hyperparameter Optimization**: Random search, TPE, search spaces
4. **Model Registry**: Versioning, staging, lifecycle management

**Key Takeaways**:
- Training infrastructure standardizes and automates the training loop
- Experiment tracking provides reproducibility and comparison of runs
- Hyperparameter optimization finds optimal configurations automatically
- Model registry manages production deployment lifecycle

**Next Steps**:
- Add distributed training support
- Implement more advanced HPO algorithms (Bayesian optimization)
- Add model serving capabilities
- Integrate with cloud storage backends
