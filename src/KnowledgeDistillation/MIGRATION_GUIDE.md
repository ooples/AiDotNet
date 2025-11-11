# Knowledge Distillation Refactoring Migration Guide

## Overview

This guide helps you migrate from the old knowledge distillation architecture to the new, refactored version that follows SOLID principles and separates concerns properly.

## What Changed?

### Summary of Changes

1. **ITeacherModel Interface Simplified**: Removed unused methods and variance modifiers
2. **Teacher Models Simplified**: Teachers now only provide logits (predictions)
3. **Adaptive Logic Moved**: Dynamic temperature adjustment moved to `AdaptiveDistillationStrategy`
4. **Curriculum Logic Moved**: Progressive difficulty adjustment moved to `CurriculumDistillationStrategy`
5. **Backward Compatibility Maintained**: Old teacher classes still work but are now simple wrappers

### Architectural Principles

The refactoring enforces proper **Separation of Concerns**:

- **Teachers**: Inference layer - only generate predictions (logits)
- **Strategies**: Training layer - handle temperature scaling, loss computation, gradient calculation

This follows the **Single Responsibility Principle** (SRP):
- Teachers are responsible for prediction
- Strategies are responsible for training logic

## Detailed Changes

### 1. ITeacherModel Interface

**Before:**
```csharp
public interface ITeacherModel<in TInput, out TOutput>
{
    TOutput GetLogits(TInput input);
    TOutput GetSoftPredictions(TInput input, double temperature);
    object? GetFeatures(TInput input, string layerName);
    object? GetAttentionWeights(TInput input);
    int OutputDimension { get; }
}
```

**After:**
```csharp
public interface ITeacherModel<TInput, TOutput>
{
    TOutput GetLogits(TInput input);
    int OutputDimension { get; }
}
```

**Why**: The removed methods were never called in the codebase. `GetSoftPredictions` duplicates strategy responsibility, while `GetFeatures` and `GetAttentionWeights` returned type-unsafe `object?`.

### 2. AdaptiveTeacherModel

**Before** (300+ lines with adaptive logic):
```csharp
public class AdaptiveTeacherModel<T>
{
    private Dictionary<int, double> _performanceHistory;
    private double _minTemperature;
    private double _maxTemperature;

    public Vector<T> GetSoftPredictions(Vector<T> input, double temperature)
    {
        // Dynamic temperature adjustment based on student performance
        double adaptiveTemp = ComputeAdaptiveTemperature(input);
        return ApplySoftmax(GetLogits(input), adaptiveTemp);
    }

    private double ComputeAdaptiveTemperature(Vector<T> input) { ... }
    public void UpdatePerformance(int sampleIndex, double performance) { ... }
}
```

**After** (simple wrapper):
```csharp
public class AdaptiveTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _baseTeacher;

    public override Vector<T> GetLogits(Vector<T> input) => _baseTeacher.GetLogits(input);
    public override int OutputDimension => _baseTeacher.OutputDimension;
}
```

**Migration**: Use `AdaptiveDistillationStrategy<T>` instead (see examples below).

### 3. CurriculumTeacherModel

**Before** (had curriculum parameters):
```csharp
public class CurriculumTeacherModel<T>
{
    private int _currentEpoch;
    private CurriculumStrategy _strategy;

    // Curriculum logic mixed into teacher
}
```

**After** (simple wrapper):
```csharp
public class CurriculumTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _baseTeacher;

    public override Vector<T> GetLogits(Vector<T> input) => _baseTeacher.GetLogits(input);
}
```

**Migration**: Use `CurriculumDistillationStrategy<T>` instead (see examples below).

## Migration Examples

### Example 1: Migrating Adaptive Distillation

**Old Code:**
```csharp
// Create adaptive teacher
var adaptiveTeacher = new AdaptiveTeacherModel<double>(
    baseTeacher,
    AdaptiveStrategy.ConfidenceBased,
    minTemperature: 1.0,
    maxTemperature: 5.0
);

// Training loop
for (int i = 0; i < samples.Length; i++)
{
    var teacherPrediction = adaptiveTeacher.GetSoftPredictions(samples[i], temperature: 3.0);
    var studentPrediction = student.Predict(samples[i]);

    // Update adaptive teacher's performance tracking
    adaptiveTeacher.UpdatePerformance(i, studentPrediction, labels[i]);

    // Compute loss manually
    var loss = ComputeKLDivergence(teacherPrediction, studentPrediction);
}
```

**New Code:**
```csharp
// Create simple teacher (just provides logits)
var teacher = new TeacherModelWrapper<double>(baseModel);

// Create adaptive strategy (handles temperature adaptation)
var strategy = new AdaptiveDistillationStrategy<double>(
    baseTemperature: 3.0,
    alpha: 0.3,
    strategy: AdaptiveStrategy.ConfidenceBased,
    minTemperature: 1.0,
    maxTemperature: 5.0,
    adaptationRate: 0.1
);

// Training loop
for (int i = 0; i < samples.Length; i++)
{
    var teacherLogits = teacher.GetLogits(samples[i]);
    var studentLogits = student.Predict(samples[i]); // Raw logits, not softmax

    // Update strategy's performance tracking
    strategy.UpdatePerformance(i, studentLogits, labels[i]);

    // Strategy handles temperature adjustment and loss computation
    var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels[i]);
    var gradient = strategy.ComputeGradient(studentLogits, teacherLogits, labels[i]);

    // Apply gradient to student
    student.ApplyGradient(gradient);
}
```

**Benefits:**
- Teacher focuses on prediction only
- Strategy handles all training logic
- Performance tracking is explicit
- Temperature adaptation is automatic per sample
- Loss and gradient computation are unified

### Example 2: Migrating Curriculum Learning

**Old Code:**
```csharp
// Create curriculum teacher
var curriculumTeacher = new CurriculumTeacherModel<double>(
    baseTeacher,
    CurriculumStrategy.EasyToHard
);

// Training loop
for (int epoch = 0; epoch < 100; epoch++)
{
    // Manually filter samples by difficulty (outside teacher)
    var currentSamples = FilterSamplesByDifficulty(samples, epoch);

    foreach (var sample in currentSamples)
    {
        var teacherPrediction = curriculumTeacher.GetSoftPredictions(sample, temperature: 3.0);
        // ... training logic
    }
}
```

**New Code:**
```csharp
// Create simple teacher
var teacher = new TeacherModelWrapper<double>(baseModel);

// Define sample difficulties (optional)
var difficulties = new Dictionary<int, double>
{
    { 0, 0.1 }, // Easy sample
    { 1, 0.3 }, // Medium sample
    { 2, 0.8 }, // Hard sample
    // ... etc
};

// Create curriculum strategy
var strategy = new CurriculumDistillationStrategy<double>(
    baseTemperature: 3.0,
    alpha: 0.3,
    strategy: CurriculumStrategy.EasyToHard,
    minTemperature: 2.0,
    maxTemperature: 5.0,
    totalSteps: 100,
    sampleDifficulties: difficulties
);

// Training loop
for (int epoch = 0; epoch < 100; epoch++)
{
    // Update curriculum progress
    strategy.UpdateProgress(epoch);

    for (int i = 0; i < samples.Length; i++)
    {
        // Strategy decides if sample should be included in current stage
        if (!strategy.ShouldIncludeSample(i))
            continue;

        var teacherLogits = teacher.GetLogits(samples[i]);
        var studentLogits = student.Predict(samples[i]);

        // Strategy automatically adjusts temperature based on curriculum stage
        var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels[i]);
        var gradient = strategy.ComputeGradient(studentLogits, teacherLogits, labels[i]);

        student.ApplyGradient(gradient);
    }
}
```

**Benefits:**
- Curriculum progression is explicit and controllable
- Sample filtering is strategy-driven
- Temperature adjusts automatically per epoch
- Supports both EasyToHard and HardToEasy
- Can optionally use difficulty scores per sample

### Example 3: Using Standard Distillation (No Changes Needed)

**Old Code (still works):**
```csharp
var teacher = new TeacherModelWrapper<double>(pretrainedModel);
var strategy = new StandardDistillationStrategy<double>(temperature: 3.0, alpha: 0.3);

// Training loop
foreach (var sample in samples)
{
    var teacherLogits = teacher.GetLogits(sample);
    var studentLogits = student.Predict(sample);
    var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels);
}
```

**New Code (identical):**
```csharp
// No changes needed - standard distillation API unchanged
var teacher = new TeacherModelWrapper<double>(pretrainedModel);
var strategy = new StandardDistillationStrategy<double>(temperature: 3.0, alpha: 0.3);

// Training loop remains the same
```

## Backward Compatibility

### What Still Works

1. **TeacherModelFactory**: All factory methods still work
   ```csharp
   var teacher = TeacherModelFactory<double>.CreateTeacher(
       TeacherModelType.Adaptive,
       model: pretrainedModel
   );
   ```

2. **AdaptiveTeacherModel/CurriculumTeacherModel**: Can still be instantiated, but are now simple wrappers
   ```csharp
   var adaptiveTeacher = new AdaptiveTeacherModel<double>(baseTeacher);
   // Works, but contains no adaptive logic - use AdaptiveDistillationStrategy instead
   ```

3. **Standard Distillation**: No changes to existing standard distillation code

### What Changed

1. **No more GetSoftPredictions**: Use `strategy.ComputeLoss()` instead of manually computing softmax
2. **No more performance tracking in teachers**: Use `strategy.UpdatePerformance()` instead
3. **No more temperature parameters in teachers**: Temperature is strategy responsibility

## Strategy Comparison

| Feature | StandardDistillationStrategy | AdaptiveDistillationStrategy | CurriculumDistillationStrategy |
|---------|------------------------------|------------------------------|--------------------------------|
| **Temperature** | Fixed | Per-sample adaptive | Per-epoch progressive |
| **Performance Tracking** | No | Yes (EMA) | Optional (via difficulties) |
| **Sample Filtering** | No | No | Yes (ShouldIncludeSample) |
| **Use Case** | Standard KD | Varies by sample confidence | Progressive difficulty |
| **Configuration** | Temperature, alpha | Min/max temp, strategy type | Curriculum direction, total steps |

## Best Practices

### 1. Choose the Right Strategy

- **StandardDistillationStrategy**: Use for basic knowledge distillation with fixed temperature
- **AdaptiveDistillationStrategy**: Use when samples vary in difficulty and student performance is uneven
- **CurriculumDistillationStrategy**: Use for structured learning progression over epochs

### 2. Temperature Range Selection

```csharp
// For adaptive strategies
minTemperature: 1.0,   // Sharp for hard/confident samples
maxTemperature: 5.0    // Soft for easy/uncertain samples

// For curriculum strategies (EasyToHard)
minTemperature: 2.0,   // End temperature (sharp, challenging)
maxTemperature: 5.0    // Start temperature (soft, gentle)
```

### 3. Alpha Tuning

```csharp
alpha: 0.3   // 30% hard loss (true labels), 70% soft loss (teacher)
alpha: 0.5   // Equal weight
alpha: 0.1   // Focus more on teacher knowledge
```

### 4. Adaptive Strategy Selection

```csharp
AdaptiveStrategy.ConfidenceBased  // Best for most cases - uses max probability
AdaptiveStrategy.EntropyBased     // Good for uncertainty-aware adaptation
AdaptiveStrategy.AccuracyBased    // Requires true labels, tracks correctness
```

## Common Migration Issues

### Issue 1: Compilation Error - GetSoftPredictions Not Found

**Error:**
```
'ITeacherModel<Vector<double>, Vector<double>>' does not contain a definition for 'GetSoftPredictions'
```

**Fix:**
Replace direct softmax calls with strategy-based loss computation:
```csharp
// OLD: var softPredictions = teacher.GetSoftPredictions(input, temperature);
// NEW: var loss = strategy.ComputeLoss(studentLogits, teacherLogits);
```

### Issue 2: AdaptiveStrategy Enum Not Found

**Error:**
```
The type or namespace name 'AdaptiveStrategy' could not be found
```

**Fix:**
This enum moved from Teachers namespace to Strategies namespace:
```csharp
// Add using directive
using AiDotNet.KnowledgeDistillation.Strategies;

// Use the strategy
var strategy = new AdaptiveDistillationStrategy<double>(
    strategy: AdaptiveStrategy.ConfidenceBased
);
```

### Issue 3: Teacher Doesn't Track Performance

**Error:**
```
'AdaptiveTeacherModel<double>' does not contain a definition for 'UpdatePerformance'
```

**Fix:**
Performance tracking moved to strategy:
```csharp
// OLD: adaptiveTeacher.UpdatePerformance(sampleIndex, studentOutput, label);
// NEW: adaptiveStrategy.UpdatePerformance(sampleIndex, studentOutput, label);
```

## FAQs

### Q: Why were these changes made?

**A**: To enforce SOLID principles:
- **Single Responsibility**: Teachers predict, strategies train
- **Separation of Concerns**: Inference logic separate from training logic
- **Interface Segregation**: Removed unused methods with type-unsafe signatures

### Q: Is the old code broken?

**A**: No. Backward compatibility is maintained. Old teacher classes still work as simple wrappers.

### Q: Do I need to migrate immediately?

**A**: No. Old code continues to work. However, using the new strategies provides:
- Better separation of concerns
- More flexible configuration
- Clearer code intent
- Production-ready implementation

### Q: What if I need custom adaptive logic?

**A**: Extend `DistillationStrategyBase<T, Vector<T>>` and implement your custom `ComputeLoss` and `ComputeGradient` methods.

### Q: Can I mix strategies?

**A**: No. Each training run should use one strategy. However, you can switch strategies between training sessions.

## Additional Resources

- **AdaptiveDistillationStrategy.cs**: See inline documentation for detailed examples
- **CurriculumDistillationStrategy.cs**: See inline documentation for curriculum learning patterns
- **DistillationStrategyBase.cs**: Base class for implementing custom strategies

## Summary

The refactoring:
- ✅ Enforces SOLID principles
- ✅ Separates inference from training logic
- ✅ Maintains backward compatibility
- ✅ Provides production-ready adaptive and curriculum strategies
- ✅ Improves code clarity and maintainability

**Migration is optional but recommended** for new code to leverage the improved architecture.
