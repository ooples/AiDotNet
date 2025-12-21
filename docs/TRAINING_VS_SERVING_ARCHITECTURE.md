# AiDotNet Architecture: Training Infrastructure vs Serving Infrastructure

## Overview

AiDotNet has two distinct but complementary infrastructure layers:

1. **Training Infrastructure** (in `src/`) - For model development, experimentation, and MLOps
2. **Serving Infrastructure** (in `src/AiDotNet.Serving/`) - For production inference deployment

---

## Layer Comparison

### AiDotNet.Serving (Inference/Production Layer)

Located in: `src/AiDotNet.Serving/`

| Component | Purpose |
|-----------|---------|
| `IModelRepository` / `ModelRepository` | Runtime model loading - loads trained models into memory for serving |
| `IRequestBatcher` / `RequestBatcher` | Batches concurrent inference requests for throughput optimization |
| `PerformanceMetrics` | Tracks inference latency (P50, P95, P99), throughput, batch utilization |
| `IServableModel<T>` | Interface for models that can be served via REST API |
| `InferenceController` | REST endpoint for predictions (`POST /api/inference/predict/{modelName}`) |
| `ModelsController` | REST endpoint for model management (load, list, unload) |

**Focus**: Production deployment, real-time inference, request batching, latency optimization

---

### AiDotNet Training Infrastructure (Development/MLOps Layer)

Located in: `src/` (various folders)

| Component | Purpose |
|-----------|---------|
| `IExperimentTracker` / `ExperimentTrackerBase` | Organizes ML experiments and training runs (like MLflow) |
| `IHyperparameterOptimizer` / `HyperparameterOptimizerBase` | Searches for optimal hyperparameters (Grid, Random, Bayesian) |
| `ICheckpointManager` / `CheckpointManagerBase` | Saves/restores training state, manages best model checkpoints |
| `ITrainingMonitor` / `TrainingMonitorBase` | Tracks training metrics (loss curves, accuracy) during model development |
| `IModelRegistry` / `ModelRegistryBase` | Model versioning and lifecycle (Development -> Staging -> Production -> Archived) |
| `IDataVersionControl` / `DataVersionControlBase` | Dataset versioning for reproducibility |

**Focus**: Model development, experimentation, reproducibility, MLOps lifecycle

---

## Key Differences

| Aspect | Training Infrastructure | Serving Infrastructure |
|--------|------------------------|----------------------|
| **When Used** | During model development | After model is trained and deployed |
| **Primary Goal** | Train and version models | Serve predictions efficiently |
| **Model Storage** | Persistent (disk-based versioning) | Runtime (in-memory for inference) |
| **Metrics Tracked** | Training loss, validation accuracy, epochs | Inference latency, throughput, batch size |
| **Lifecycle Stage** | Development, experimentation | Production deployment |

---

## Integration Flow

```
+-------------------+     +------------------+     +----------------------+
|  Training Layer   | --> |  Model Registry  | --> |   Serving Layer      |
+-------------------+     +------------------+     +----------------------+
                          |                        |
| ExperimentTracker |     | Version 1 (Dev)  |     | ModelRepository      |
| CheckpointManager |     | Version 2 (Stage)|     | (loads v3 into RAM)  |
| TrainingMonitor   |     | Version 3 (Prod) | --> |                      |
| HyperparamOptim   |     | Version 4 (Arch) |     | RequestBatcher       |
| DataVersionControl|     +------------------+     | PerformanceMetrics   |
+-------------------+                              +----------------------+
```

### Workflow Example

1. **Experiment Phase** (Training Infrastructure)
   - Create experiment with `ExperimentTracker`
   - Track training with `TrainingMonitor`
   - Save checkpoints with `CheckpointManager`
   - Optimize hyperparameters with `HyperparameterOptimizer`
   - Version datasets with `DataVersionControl`

2. **Registration Phase** (Model Registry)
   - Register trained model: `ModelRegistry.RegisterModel()`
   - Create versions: `ModelRegistry.CreateModelVersion()`
   - Promote through stages: Development -> Staging -> Production

3. **Deployment Phase** (Serving Infrastructure)
   - Load from registry: `ModelRepository.LoadModel()`
   - Serve predictions via REST API
   - Monitor inference with `PerformanceMetrics`

---

## Why Both Layers Are Needed

### Model Registry (Training) vs Model Repository (Serving)

| ModelRegistry (Training) | ModelRepository (Serving) |
|--------------------------|---------------------------|
| Persistent storage on disk | In-memory for fast inference |
| Multiple versions per model | Single loaded instance per name |
| Lifecycle stages (Dev/Staging/Prod) | Just "loaded" or "not loaded" |
| Metadata, lineage, comparison | Runtime info (when loaded, dimensions) |
| Used during development | Used in production |

### Training Monitor vs Performance Metrics

| TrainingMonitor (Training) | PerformanceMetrics (Serving) |
|----------------------------|------------------------------|
| Loss curves over epochs | Latency percentiles (P50, P95, P99) |
| Validation metrics | Throughput (requests/second) |
| Resource usage during training | Batch utilization efficiency |
| Epoch start/end events | Queue depth monitoring |
| Used during model training | Used during production inference |

---

## Folder Structure

```
src/
├── ExperimentTracking/
│   ├── ExperimentTrackerBase.cs      # Base class
│   └── ExperimentTracker.cs          # Implementation
├── HyperparameterOptimization/
│   ├── HyperparameterOptimizerBase.cs
│   ├── RandomSearchOptimizer.cs
│   └── GridSearchOptimizer.cs
├── CheckpointManagement/
│   ├── CheckpointManagerBase.cs
│   └── CheckpointManager.cs
├── TrainingMonitoring/
│   ├── TrainingMonitorBase.cs
│   └── TrainingMonitor.cs            # To be implemented
├── ModelRegistry/
│   ├── ModelRegistryBase.cs
│   └── ModelRegistry.cs              # To be implemented
├── DataVersionControl/
│   ├── DataVersionControlBase.cs
│   └── DataVersionControl.cs         # To be implemented
│
└── AiDotNet.Serving/                 # Separate project for serving
    ├── Services/
    │   ├── IModelRepository.cs
    │   ├── ModelRepository.cs
    │   ├── IRequestBatcher.cs
    │   └── RequestBatcher.cs
    ├── Monitoring/
    │   └── PerformanceMetrics.cs
    └── Controllers/
        ├── InferenceController.cs
        └── ModelsController.cs
```

---

## Summary

- **Training Infrastructure**: Build, experiment, version, and manage models during development
- **Serving Infrastructure**: Deploy and serve models efficiently in production
- **Integration Point**: `ModelRegistry` stores versioned models that `ModelRepository` loads for serving
- **No Overlap**: Each layer serves a distinct phase of the ML lifecycle
