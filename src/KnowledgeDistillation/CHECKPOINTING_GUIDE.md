# Knowledge Distillation Checkpointing Guide

## Overview

This guide explains how to use checkpointing during knowledge distillation training, including saving/loading student models, managing curriculum progress, and supporting multi-stage distillation.

## Why Checkpointing Matters for Distillation

### 1. Training Resumption
Distillation training can take hours or days. Checkpointing allows you to:
- Resume after interruptions (power loss, system crashes)
- Pause training and continue later
- Save compute resources by not restarting from scratch

### 2. Best Model Selection
Training doesn't always improve monotonically. Checkpointing helps:
- Save the student model with best validation performance
- Roll back if student starts overfitting
- Compare performance at different training stages

### 3. Multi-Stage Distillation
Progressive compression requires checkpointing:
- Stage 1: Large teacher → Medium student
- Stage 2: Medium student (becomes teacher) → Small student
- Stage 3: Small student → Tiny student

### 4. Curriculum Learning State
Resume curriculum learning from correct stage:
- Save curriculum progress with checkpoint
- Don't restart from "easy samples" after interruption
- Maintain temperature progression

### 5. Experiment Tracking
Compare different approaches:
- Save checkpoints for different strategies
- Track performance metrics over time
- Debug training dynamics

## Architecture

### Interfaces

```
ICheckpointableModel (stream-based, flexible)
├─ SaveState(Stream stream)
└─ LoadState(Stream stream)

IModelSerializer (file-based, convenience)
├─ SaveModel(string filePath)
└─ LoadModel(string filePath)
```

### Checkpoint Manager

```
DistillationCheckpointManager<T>
├─ DistillationCheckpointConfig (configuration)
├─ SaveCheckpointIfNeeded() (automatic saving)
├─ LoadBestCheckpoint() (load by metric)
└─ GetBestCheckpoint() (query metadata)
```

## Quick Start: Automatic Checkpointing (Recommended)

The easiest way to enable checkpointing is through automatic checkpointing built into the trainer. Simply configure the checkpoint settings and the trainer handles everything automatically.

### Automatic Checkpointing Example

```csharp
using AiDotNet.KnowledgeDistillation;

// Create trainer
var teacher = LoadPretrainedTeacher();
var student = CreateStudentModel();  // Must implement ICheckpointableModel
var strategy = new ConfidenceBasedAdaptiveStrategy<double>();

var trainer = new KnowledgeDistillationTrainer<double, Vector<double>, Vector<double>>(
    teacher,
    strategy
);

// Enable automatic checkpointing by setting CheckpointConfig
trainer.CheckpointConfig = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./checkpoints",
    SaveEveryEpochs = 5,          // Auto-save every 5 epochs
    KeepBestN = 3,                // Keep only 3 best checkpoints
    SaveStudent = true,
    BestMetric = "validation_loss",
    LowerIsBetter = true
};

// Set the student model (required for checkpointing)
trainer.Student = student as ICheckpointableModel;

// Train - checkpointing happens automatically!
trainer.Train(
    studentForward: student.Predict,
    studentBackward: student.ApplyGradient,
    trainInputs: trainingData,
    trainLabels: trainingLabels,
    epochs: 100,
    batchSize: 32,
    validationInputs: validationData,
    validationLabels: validationLabels
);

// After training completes, the best checkpoint is automatically loaded!
Console.WriteLine("Training complete. Best checkpoint automatically restored.");
```

**What happens automatically:**
1. **OnTrainingStart**: Checkpoint manager is initialized
2. **OnEpochEnd**: Checkpoints are saved based on your configuration
3. **OnValidationComplete**: Validation metrics are tracked for best checkpoint selection
4. **OnTrainingEnd**: Best checkpoint is automatically loaded

**Benefits:**
- ✅ Zero manual checkpoint management code
- ✅ Automatic best model selection
- ✅ Automatic checkpoint pruning (keeps only best N)
- ✅ Curriculum state preservation (if using curriculum strategies)
- ✅ Clean, simple API

### Disabling Automatic Checkpointing

```csharp
// Default: no checkpointing
trainer.CheckpointConfig = null;  // or simply don't set it

// Training proceeds without checkpointing
trainer.Train(...);
```

## Manual Checkpointing (Advanced)

For advanced use cases where you need fine-grained control over checkpoint timing and logic, you can use the `DistillationCheckpointManager` directly.

### Example 1: Simple Student Checkpointing (Manual)

```csharp
using AiDotNet.KnowledgeDistillation;
using AiDotNet.Interfaces;

// Configure checkpointing
var checkpointConfig = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./checkpoints",
    SaveEveryEpochs = 5,          // Save every 5 epochs
    KeepBestN = 3,                // Keep only 3 best checkpoints
    SaveStudent = true,
    SaveTeacher = false,          // Don't save teacher (already trained)
    BestMetric = "validation_loss",
    LowerIsBetter = true          // Lower loss is better
};

var checkpointManager = new DistillationCheckpointManager<double>(checkpointConfig);

// Training loop
var teacher = LoadPretrainedTeacher();
var student = CreateStudentModel();
var strategy = new ConfidenceBasedAdaptiveStrategy<double>();

for (int epoch = 0; epoch < 100; epoch++)
{
    // Train student for one epoch
    for (int batch = 0; batch < numBatches; batch++)
    {
        var teacherLogits = teacher.GetLogits(samples[batch]);
        var studentLogits = student.Predict(samples[batch]);

        var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels[batch]);
        var gradient = strategy.ComputeGradient(studentLogits, teacherLogits, labels[batch]);

        student.ApplyGradient(gradient);
    }

    // Evaluate on validation set
    double validationLoss = EvaluateOnValidationSet(student);
    Console.WriteLine($"Epoch {epoch}: Validation Loss = {validationLoss}");

    // Save checkpoint if needed
    var metrics = new Dictionary<string, double>
    {
        { "validation_loss", validationLoss },
        { "training_loss", trainingLoss }
    };

    bool saved = checkpointManager.SaveCheckpointIfNeeded(
        epoch: epoch,
        student: student as ICheckpointableModel,  // Student must implement ICheckpointableModel
        metrics: metrics
    );

    if (saved)
    {
        Console.WriteLine($"Checkpoint saved at epoch {epoch}");
    }
}

// After training, load the best checkpoint
Console.WriteLine("Loading best checkpoint...");
var bestCheckpoint = checkpointManager.LoadBestCheckpoint(student as ICheckpointableModel);
if (bestCheckpoint != null)
{
    Console.WriteLine($"Best checkpoint from epoch {bestCheckpoint.Epoch}");
    Console.WriteLine($"Validation loss: {bestCheckpoint.Metrics["validation_loss"]}");
}
```

### Example 2: Curriculum Learning with Checkpointing

```csharp
// Configure checkpointing for curriculum learning
var checkpointConfig = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./curriculum_checkpoints",
    SaveEveryEpochs = 10,
    KeepBestN = 5,
    SaveStudent = true,
    SaveCurriculumState = true,     // Save curriculum progress!
    BestMetric = "validation_accuracy",
    LowerIsBetter = false            // Higher accuracy is better
};

var checkpointManager = new DistillationCheckpointManager<double>(checkpointConfig);

// Set up curriculum strategy
var difficulties = ComputeSampleDifficulties(trainingSamples);
var curriculumStrategy = new EasyToHardCurriculumStrategy<double>(
    minTemperature: 2.0,
    maxTemperature: 5.0,
    totalSteps: 100,
    sampleDifficulties: difficulties
);

// Training loop with curriculum
for (int epoch = 0; epoch < 100; epoch++)
{
    // Update curriculum progress
    curriculumStrategy.UpdateProgress(epoch);
    Console.WriteLine($"Curriculum progress: {curriculumStrategy.CurriculumProgress:P0}");

    // Train on samples appropriate for current curriculum stage
    foreach (var (sample, index) in trainingSamples.WithIndex())
    {
        // Filter by curriculum
        if (!curriculumStrategy.ShouldIncludeSample(index))
            continue;

        var teacherLogits = teacher.GetLogits(sample);
        var studentLogits = student.Predict(sample);

        var loss = curriculumStrategy.ComputeLoss(studentLogits, teacherLogits, labels[index]);
        student.ApplyGradient(curriculumStrategy.ComputeGradient(studentLogits, teacherLogits, labels[index]));
    }

    // Evaluate and checkpoint
    double accuracy = EvaluateAccuracy(student);
    checkpointManager.SaveCheckpointIfNeeded(
        epoch: epoch,
        student: student as ICheckpointableModel,
        strategy: curriculumStrategy,  // Save curriculum state!
        metrics: new Dictionary<string, double> { { "validation_accuracy", accuracy } }
    );
}
```

### Example 3: Multi-Stage Distillation

```csharp
// Stage 1: Large teacher → Medium student
Console.WriteLine("Stage 1: Large → Medium");

var largeTeacher = LoadPretrainedLargeModel();
var mediumStudent = CreateMediumModel();

var stage1Config = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./stage1_checkpoints",
    SaveEveryEpochs = 10,
    KeepBestN = 1,  // Keep only best
    SaveStudent = true,
    SaveTeacher = false
};

var stage1Manager = new DistillationCheckpointManager<double>(stage1Config);
var stage1Strategy = new ConfidenceBasedAdaptiveStrategy<double>();

// Train medium student
for (int epoch = 0; epoch < 50; epoch++)
{
    TrainForOneEpoch(mediumStudent, largeTeacher, stage1Strategy);

    double loss = EvaluateOnValidationSet(mediumStudent);
    stage1Manager.SaveCheckpointIfNeeded(
        epoch: epoch,
        student: mediumStudent as ICheckpointableModel,
        metrics: new Dictionary<string, double> { { "validation_loss", loss } }
    );
}

// Load best medium model
stage1Manager.LoadBestCheckpoint(mediumStudent as ICheckpointableModel);

// Stage 2: Medium student (now teacher) → Small student
Console.WriteLine("Stage 2: Medium → Small");

// Medium student becomes teacher for next stage!
var mediumTeacher = new TeacherModelWrapper<double>(mediumStudent);
var smallStudent = CreateSmallModel();

var stage2Config = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./stage2_checkpoints",
    SaveEveryEpochs = 10,
    KeepBestN = 1,
    SaveStudent = true,
    SaveTeacher = true  // Save medium teacher in case we need it
};

var stage2Manager = new DistillationCheckpointManager<double>(stage2Config);
var stage2Strategy = new ConfidenceBasedAdaptiveStrategy<double>();

// Train small student
for (int epoch = 0; epoch < 50; epoch++)
{
    TrainForOneEpoch(smallStudent, mediumTeacher, stage2Strategy);

    double loss = EvaluateOnValidationSet(smallStudent);
    stage2Manager.SaveCheckpointIfNeeded(
        epoch: epoch,
        student: smallStudent as ICheckpointableModel,
        teacher: mediumTeacher as ICheckpointableModel,
        metrics: new Dictionary<string, double> { { "validation_loss", loss } }
    );
}

// Load best small model
stage2Manager.LoadBestCheckpoint(smallStudent as ICheckpointableModel);

Console.WriteLine("Multi-stage distillation complete!");
Console.WriteLine($"Final model size: {smallStudent.ParameterCount} parameters");
```

### Example 4: Resuming Interrupted Training

```csharp
// Try to resume from existing checkpoint, or start fresh
var checkpointConfig = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./checkpoints",
    SaveEveryEpochs = 5,
    KeepBestN = 3
};

var checkpointManager = new DistillationCheckpointManager<double>(checkpointConfig);

var student = CreateStudentModel();
int startEpoch = 0;

// Try to load most recent checkpoint
var recentCheckpoint = checkpointManager.SavedCheckpoints
    .OrderByDescending(c => c.Epoch)
    .FirstOrDefault();

if (recentCheckpoint != null)
{
    Console.WriteLine($"Resuming from epoch {recentCheckpoint.Epoch}");
    checkpointManager.LoadCheckpoint(recentCheckpoint, student as ICheckpointableModel);
    startEpoch = recentCheckpoint.Epoch + 1;
}
else
{
    Console.WriteLine("Starting training from scratch");
}

// Continue training
for (int epoch = startEpoch; epoch < 100; epoch++)
{
    // ... training code ...

    checkpointManager.SaveCheckpointIfNeeded(
        epoch: epoch,
        student: student as ICheckpointableModel,
        metrics: new Dictionary<string, double> { { "validation_loss", validationLoss } }
    );
}
```

### Example 5: Batch-Level Checkpointing (Long Epochs)

```csharp
var checkpointConfig = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./batch_checkpoints",
    SaveEveryEpochs = 0,       // Disable epoch-based
    SaveEveryBatches = 1000,   // Save every 1000 batches
    KeepBestN = 5
};

var checkpointManager = new DistillationCheckpointManager<double>(checkpointConfig);

for (int epoch = 0; epoch < 10; epoch++)
{
    for (int batch = 0; batch < 10000; batch++)  // Very long epoch
    {
        // Training step
        var teacherLogits = teacher.GetLogits(samples[batch]);
        var studentLogits = student.Predict(samples[batch]);

        var loss = strategy.ComputeLoss(studentLogits, teacherLogits);
        student.ApplyGradient(strategy.ComputeGradient(studentLogits, teacherLogits));

        // Checkpoint every 1000 batches
        if ((batch + 1) % 100 == 0)  // Validate every 100 batches
        {
            double validationLoss = QuickValidation(student);

            checkpointManager.SaveCheckpointIfNeeded(
                epoch: epoch,
                batch: batch,
                student: student as ICheckpointableModel,
                metrics: new Dictionary<string, double> { { "validation_loss", validationLoss } }
            );
        }
    }
}
```

## Best Practices

### 1. Choose Appropriate Save Frequency

**Too Frequent:**
- Wastes disk space
- Slows down training
- Makes it hard to find good checkpoints

**Too Infrequent:**
- Risk losing significant progress
- Miss best model if it overfits quickly

**Recommended:**
- Short training (<50 epochs): Save every 5-10 epochs
- Long training (>100 epochs): Save every 10-20 epochs
- Very long epochs: Use batch-level checkpointing

### 2. Monitor Multiple Metrics

```csharp
var metrics = new Dictionary<string, double>
{
    { "validation_loss", validationLoss },
    { "validation_accuracy", validationAccuracy },
    { "training_loss", trainingLoss },
    { "distillation_temperature", strategy.ComputeAdaptiveTemperature(...) }
};
```

### 3. Keep Best Checkpoints Only

```csharp
var checkpointConfig = new DistillationCheckpointConfig
{
    KeepBestN = 3,  // Keep top 3 checkpoints only
    BestMetric = "validation_loss",
    LowerIsBetter = true
};
```

This prevents disk space issues while ensuring you have the best models.

### 4. Save Teacher for Multi-Stage Distillation

```csharp
var checkpointConfig = new DistillationCheckpointConfig
{
    SaveStudent = true,
    SaveTeacher = true,  // If student will become teacher later
    SaveCurriculumState = true  // If using curriculum learning
};
```

### 5. Use Descriptive Checkpoint Directories

```csharp
var config1 = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./experiments/distillation_confidence_lr0.001"
};

var config2 = new DistillationCheckpointConfig
{
    CheckpointDirectory = "./experiments/distillation_entropy_lr0.01"
};
```

## Implementing ICheckpointableModel

If your model doesn't implement `ICheckpointableModel`, add it:

```csharp
public class MyStudentModel : ICheckpointableModel
{
    private double[] _weights;
    private double[] _biases;

    public void SaveState(Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Write weights
        writer.Write(_weights.Length);
        foreach (var weight in _weights)
        {
            writer.Write(weight);
        }

        // Write biases
        writer.Write(_biases.Length);
        foreach (var bias in _biases)
        {
            writer.Write(bias);
        }
    }

    public void LoadState(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // Read weights
        int weightCount = reader.ReadInt32();
        _weights = new double[weightCount];
        for (int i = 0; i < weightCount; i++)
        {
            _weights[i] = reader.ReadDouble();
        }

        // Read biases
        int biasCount = reader.ReadInt32();
        _biases = new double[biasCount];
        for (int i = 0; i < biasCount; i++)
        {
            _biases[i] = reader.ReadDouble();
        }
    }
}
```

## Common Patterns

### Pattern 1: Early Stopping with Patience

```csharp
int patience = 10;
int epochsWithoutImprovement = 0;
double bestLoss = double.MaxValue;

for (int epoch = 0; epoch < maxEpochs; epoch++)
{
    // Training...

    double validationLoss = EvaluateOnValidationSet(student);

    if (validationLoss < bestLoss)
    {
        bestLoss = validationLoss;
        epochsWithoutImprovement = 0;

        // Force save when we find new best
        checkpointManager.SaveCheckpointIfNeeded(
            epoch: epoch,
            student: student as ICheckpointableModel,
            metrics: new Dictionary<string, double> { { "validation_loss", validationLoss } },
            force: true  // Force save
        );
    }
    else
    {
        epochsWithoutImprovement++;

        if (epochsWithoutImprovement >= patience)
        {
            Console.WriteLine($"Early stopping at epoch {epoch}");
            break;
        }
    }
}

// Load best model
checkpointManager.LoadBestCheckpoint(student as ICheckpointableModel);
```

### Pattern 2: Comparing Multiple Strategies

```csharp
var strategies = new[]
{
    ("confidence", new ConfidenceBasedAdaptiveStrategy<double>()),
    ("accuracy", new AccuracyBasedAdaptiveStrategy<double>()),
    ("entropy", new EntropyBasedAdaptiveStrategy<double>())
};

foreach (var (name, strategy) in strategies)
{
    var student = CreateStudentModel();

    var config = new DistillationCheckpointConfig
    {
        CheckpointDirectory = $"./experiments/{name}",
        CheckpointPrefix = name,
        SaveEveryEpochs = 10,
        KeepBestN = 1
    };

    var manager = new DistillationCheckpointManager<double>(config);

    // Train with this strategy
    TrainWithStrategy(student, teacher, strategy, manager);

    // Load best and evaluate
    manager.LoadBestCheckpoint(student as ICheckpointableModel);
    double finalAccuracy = EvaluateFinalAccuracy(student);

    Console.WriteLine($"{name}: Final accuracy = {finalAccuracy:P2}");
}
```

## Summary

Checkpointing in knowledge distillation:
- ✅ Use `DistillationCheckpointManager<T>` for automatic checkpoint management
- ✅ Models must implement `ICheckpointableModel` for checkpointing
- ✅ Save curriculum state for curriculum learning strategies
- ✅ Use multi-stage distillation for progressive compression
- ✅ Monitor multiple metrics and keep best checkpoints only
- ✅ Resume training after interruptions
- ✅ Compare different strategies through checkpointing

**Key Takeaway**: Checkpointing is essential for production knowledge distillation pipelines!
