# Knowledge Distillation Refactoring Migration Guide

## Overview

This guide helps you migrate to the refactored knowledge distillation architecture that follows SOLID principles with proper interfaces, abstract base classes, and concrete implementations.

## What Changed?

### Summary of Changes

1. **ITeacherModel Interface Simplified**: Removed unused methods and variance modifiers
2. **Teacher Models Simplified**: Teachers now only provide logits (predictions)
3. **Adaptive Strategies Improved**: Created proper interface hierarchy with 3 concrete strategies
4. **Curriculum Strategies Improved**: Created proper interface hierarchy with 2 concrete strategies
5. **Open/Closed Principle**: Easy to extend with new strategies without modifying existing code

### New Architecture

The refactoring creates a proper **Strategy Pattern** hierarchy:

```
IDistillationStrategy<T, TOutput>
│
├─ DistillationStrategyBase<T, TOutput>
│  │
│  ├─ StandardDistillationStrategy<T>
│  │
│  ├─ IAdaptiveDistillationStrategy<T>
│  │  │
│  │  ├─ AdaptiveDistillationStrategyBase<T>
│  │  │  │
│  │  │  ├─ ConfidenceBasedAdaptiveStrategy<T>
│  │  │  ├─ AccuracyBasedAdaptiveStrategy<T>
│  │  │  └─ EntropyBasedAdaptiveStrategy<T>
│  │
│  └─ ICurriculumDistillationStrategy<T>
│     │
│     ├─ CurriculumDistillationStrategyBase<T>
│        │
│        ├─ EasyToHardCurriculumStrategy<T>
│        └─ HardToEasyCurriculumStrategy<T>
```

### Architectural Principles

**Separation of Concerns**:
- **Teachers**: Inference layer - only generate predictions (logits)
- **Strategies**: Training layer - handle temperature scaling, loss computation, gradient calculation

**SOLID Principles**:
- **Single Responsibility**: Teachers predict, strategies train
- **Open/Closed**: Add new strategies without modifying existing code
- **Liskov Substitution**: All strategies are interchangeable through interfaces
- **Interface Segregation**: Specific interfaces (IAdaptive, ICurriculum) for specialized features
- **Dependency Inversion**: Depend on abstractions (interfaces), not concretions

## Migration Examples

### Example 1: Confidence-Based Adaptive Distillation

**Old Code (enum-based switching):**
```csharp
// OLD: Single class with enum parameter
var strategy = new AdaptiveDistillationStrategy<double>(
    baseTemperature: 3.0,
    alpha: 0.3,
    strategy: AdaptiveStrategy.ConfidenceBased,  // ❌ Enum switching
    minTemperature: 1.0,
    maxTemperature: 5.0
);
```

**New Code (concrete class):**
```csharp
// NEW: Specific strategy class
var strategy = new ConfidenceBasedAdaptiveStrategy<double>(
    baseTemperature: 3.0,
    alpha: 0.3,
    minTemperature: 1.0,
    maxTemperature: 5.0,
    adaptationRate: 0.1
);

// Training loop
for (int i = 0; i < samples.Length; i++)
{
    var teacherLogits = teacher.GetLogits(samples[i]);
    var studentLogits = student.Predict(samples[i]);

    // Update performance tracking
    strategy.UpdatePerformance(i, studentLogits);

    // Strategy handles adaptive temperature automatically
    var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels[i]);
    var gradient = strategy.ComputeGradient(studentLogits, teacherLogits, labels[i]);

    student.ApplyGradient(gradient);
}
```

**Benefits**:
- ✅ No enum switching (Open/Closed Principle)
- ✅ Each strategy is a dedicated class
- ✅ Easy to extend with new adaptive strategies
- ✅ Better testability (mock specific strategy)

### Example 2: Accuracy-Based Adaptive Distillation

**New Code:**
```csharp
// Tracks correctness instead of confidence
var strategy = new AccuracyBasedAdaptiveStrategy<double>(
    minTemperature: 1.5,  // For samples student gets right
    maxTemperature: 6.0,  // For samples student gets wrong
    adaptationRate: 0.2   // How fast to adapt
);

for (int i = 0; i < samples.Length; i++)
{
    var teacherLogits = teacher.GetLogits(samples[i]);
    var studentLogits = student.Predict(samples[i]);

    // IMPORTANT: Pass labels for accuracy tracking
    strategy.UpdatePerformance(i, studentLogits, labels[i]);
    var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels[i]);

    student.ApplyGradient(strategy.ComputeGradient(studentLogits, teacherLogits, labels[i]));
}
```

### Example 3: Entropy-Based Adaptive Distillation

**New Code:**
```csharp
// Adapts based on prediction uncertainty (entropy)
var strategy = new EntropyBasedAdaptiveStrategy<double>(
    minTemperature: 1.5,  // For uncertain predictions (high entropy)
    maxTemperature: 4.0,  // For confident predictions (low entropy)
    adaptationRate: 0.15
);

// Automatically adapts based on entropy - no labels needed!
for (int i = 0; i < samples.Length; i++)
{
    var teacherLogits = teacher.GetLogits(samples[i]);
    var studentLogits = student.Predict(samples[i]);

    strategy.UpdatePerformance(i, studentLogits);
    var loss = strategy.ComputeLoss(studentLogits, teacherLogits);

    student.ApplyGradient(strategy.ComputeGradient(studentLogits, teacherLogits));
}
```

### Example 4: Easy-to-Hard Curriculum Learning

**Old Code (enum-based switching):**
```csharp
// OLD: Single class with enum parameter
var strategy = new CurriculumDistillationStrategy<double>(
    strategy: CurriculumStrategy.EasyToHard,  // ❌ Enum switching
    minTemperature: 2.0,
    maxTemperature: 5.0,
    totalSteps: 100
);
```

**New Code (concrete class):**
```csharp
// NEW: Specific curriculum strategy class
var difficulties = new Dictionary<int, double>
{
    { 0, 0.1 },  // Easy sample
    { 1, 0.3 },  // Medium sample
    { 2, 0.8 },  // Hard sample
};

var strategy = new EasyToHardCurriculumStrategy<double>(
    minTemperature: 2.0,   // Final temperature (hard phase)
    maxTemperature: 5.0,   // Initial temperature (easy phase)
    totalSteps: 100,       // 100 epochs
    sampleDifficulties: difficulties
);

// Training loop with curriculum filtering
for (int epoch = 0; epoch < 100; epoch++)
{
    strategy.UpdateProgress(epoch);  // Advance curriculum

    foreach (var (sample, index) in trainingSamples.WithIndex())
    {
        // Filter samples by curriculum
        if (!strategy.ShouldIncludeSample(index))
            continue; // Too hard for current stage

        var teacherLogits = teacher.GetLogits(sample);
        var studentLogits = student.Predict(sample);

        // Temperature automatically adjusts based on epoch
        var loss = strategy.ComputeLoss(studentLogits, teacherLogits, labels[index]);
        student.ApplyGradient(strategy.ComputeGradient(studentLogits, teacherLogits, labels[index]));
    }
}
```

### Example 5: Hard-to-Easy Curriculum (Fine-Tuning)

**New Code:**
```csharp
// For fine-tuning already-trained students
var strategy = new HardToEasyCurriculumStrategy<double>(
    minTemperature: 1.5,   // Initial temperature (hard phase)
    maxTemperature: 4.0,   // Final temperature (easy phase)
    totalSteps: 50,        // Shorter for fine-tuning
    sampleDifficulties: difficulties
);

// Fine-tuning loop
for (int epoch = 0; epoch < 50; epoch++)
{
    strategy.UpdateProgress(epoch);

    foreach (var (sample, index) in trainingSamples.WithIndex())
    {
        // Filter: Only hard samples early, all samples later
        if (!strategy.ShouldIncludeSample(index))
            continue; // Too easy for current stage

        // Fine-tune on this sample...
    }
}
```

## Strategy Comparison

| Strategy | Adaptation Basis | Requires Labels | Best For |
|----------|-----------------|-----------------|----------|
| **ConfidenceBasedAdaptiveStrategy** | Max probability | No | General-purpose, varying difficulty |
| **AccuracyBasedAdaptiveStrategy** | Correctness | Yes | Supervised learning, labeled data |
| **EntropyBasedAdaptiveStrategy** | Uncertainty | No | Uncertainty-aware adaptation |
| **EasyToHardCurriculumStrategy** | Training progress | No (optional) | Training from scratch |
| **HardToEasyCurriculumStrategy** | Training progress (inverted) | No (optional) | Fine-tuning, transfer learning |

## Creating Custom Strategies

### Custom Adaptive Strategy

```csharp
public class MyCustomAdaptiveStrategy<T> : AdaptiveDistillationStrategyBase<T>
{
    public MyCustomAdaptiveStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 1.0,
        double maxTemperature = 5.0,
        double adaptationRate = 0.1)
        : base(baseTemperature, alpha, minTemperature, maxTemperature, adaptationRate)
    {
    }

    public override double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput)
    {
        // Your custom adaptation logic here
        // Example: Adapt based on teacher-student agreement
        var studentProbs = Softmax(studentOutput, 1.0);
        var teacherProbs = Softmax(teacherOutput, 1.0);

        double agreement = ComputeAgreement(studentProbs, teacherProbs);
        double difficulty = 1.0 - agreement;

        return MinTemperature + difficulty * (MaxTemperature - MinTemperature);
    }

    private double ComputeAgreement(Vector<T> studentProbs, Vector<T> teacherProbs)
    {
        // Compute cosine similarity or KL divergence
        // ...
    }
}
```

### Custom Curriculum Strategy

```csharp
public class PacedCurriculumStrategy<T> : CurriculumDistillationStrategyBase<T>
{
    private readonly double _pacingFunction;

    public PacedCurriculumStrategy(
        double baseTemperature = 3.0,
        double alpha = 0.3,
        double minTemperature = 2.0,
        double maxTemperature = 5.0,
        int totalSteps = 100,
        double pacingFunction = 2.0)  // Controls progression speed
        : base(baseTemperature, alpha, minTemperature, maxTemperature, totalSteps)
    {
        _pacingFunction = pacingFunction;
    }

    public override bool ShouldIncludeSample(int sampleIndex)
    {
        double? difficulty = GetSampleDifficulty(sampleIndex);
        if (difficulty == null) return true;

        // Non-linear pacing (e.g., exponential)
        double effectiveProgress = Math.Pow(CurriculumProgress, _pacingFunction);
        return difficulty.Value <= effectiveProgress;
    }

    public override double ComputeCurriculumTemperature()
    {
        // Non-linear temperature progression
        double effectiveProgress = Math.Pow(CurriculumProgress, _pacingFunction);
        return MaxTemperature - effectiveProgress * (MaxTemperature - MinTemperature);
    }
}
```

## Interface Reference

### IAdaptiveDistillationStrategy<T>

```csharp
public interface IAdaptiveDistillationStrategy<T>
{
    double MinTemperature { get; }
    double MaxTemperature { get; }
    double AdaptationRate { get; }

    void UpdatePerformance(int sampleIndex, Vector<T> studentOutput, Vector<T>? trueLabel = null);
    double ComputeAdaptiveTemperature(Vector<T> studentOutput, Vector<T> teacherOutput);
    double GetPerformance(int sampleIndex);
}
```

### ICurriculumDistillationStrategy<T>

```csharp
public interface ICurriculumDistillationStrategy<T>
{
    int TotalSteps { get; }
    double CurriculumProgress { get; }
    double MinTemperature { get; }
    double MaxTemperature { get; }

    void UpdateProgress(int step);
    void SetSampleDifficulty(int sampleIndex, double difficulty);
    bool ShouldIncludeSample(int sampleIndex);
    double ComputeCurriculumTemperature();
    double? GetSampleDifficulty(int sampleIndex);
}
```

## Benefits of New Architecture

### 1. Open/Closed Principle
**Before**: Adding new strategy required modifying enum and switch statement
```csharp
// ❌ Required modifying existing code
public enum AdaptiveStrategy
{
    ConfidenceBased,
    AccuracyBased,
    EntropyBased,
    MyNewStrategy  // ← Must modify enum
}

switch (_strategy)
{
    case ConfidenceBased: ...
    case AccuracyBased: ...
    case EntropyBased: ...
    case MyNewStrategy: ... // ← Must modify switch
}
```

**After**: Just create new class
```csharp
// ✅ No modification of existing code needed
public class MyNewAdaptiveStrategy<T> : AdaptiveDistillationStrategyBase<T>
{
    public override double ComputeAdaptiveTemperature(...) { ... }
}
```

### 2. Testability
**Before**: Mock entire class, test specific enum branch
```csharp
// ❌ Complex testing
var mock = new Mock<AdaptiveDistillationStrategy<double>>();
// How to test just ConfidenceBased logic?
```

**After**: Test specific strategy in isolation
```csharp
// ✅ Clean, focused testing
var strategy = new ConfidenceBasedAdaptiveStrategy<double>();
var temp = strategy.ComputeAdaptiveTemperature(studentOutput, teacherOutput);
Assert.InRange(temp, strategy.MinTemperature, strategy.MaxTemperature);
```

### 3. Composition
**Before**: Can't combine strategies
```csharp
// ❌ Can't mix ConfidenceBased + CurriculumLearning easily
```

**After**: Compose strategies through interfaces
```csharp
// ✅ Can create hybrid strategies
public class HybridStrategy<T> : AdaptiveDistillationStrategyBase<T>
{
    private readonly ICurriculumDistillationStrategy<T> _curriculum;

    public override double ComputeAdaptiveTemperature(...)
    {
        double adaptiveTemp = base.ComputeAdaptiveTemperature(...);
        double curriculumTemp = _curriculum.ComputeCurriculumTemperature();
        return (adaptiveTemp + curriculumTemp) / 2.0; // Blend both
    }
}
```

### 4. Dependency Injection
**Before**: Hard to inject strategies
```csharp
// ❌ Tightly coupled to concrete enum
```

**After**: Inject through interface
```csharp
// ✅ Flexible dependency injection
public class DistillationTrainer<T>
{
    private readonly IAdaptiveDistillationStrategy<T> _strategy;

    public DistillationTrainer(IAdaptiveDistillationStrategy<T> strategy)
    {
        _strategy = strategy; // Any adaptive strategy works!
    }
}

// Usage
var trainer1 = new DistillationTrainer(new ConfidenceBasedAdaptiveStrategy<double>());
var trainer2 = new DistillationTrainer(new MyCustomAdaptiveStrategy<double>());
```

## Common Migration Issues

### Issue 1: Enum Not Found

**Error:**
```
The type or namespace name 'AdaptiveStrategy' could not be found
```

**Fix:**
Replace enum with specific strategy class:
```csharp
// OLD:
new AdaptiveDistillationStrategy<double>(strategy: AdaptiveStrategy.ConfidenceBased)

// NEW:
new ConfidenceBasedAdaptiveStrategy<double>()
```

### Issue 2: UpdatePerformance Method Signature Changed

**Error:**
```
Cannot convert Vector<double> to double
```

**Fix:**
Pass full vector, not just a scalar:
```csharp
// OLD:
strategy.UpdatePerformance(sampleIndex, performanceScalar);

// NEW:
strategy.UpdatePerformance(sampleIndex, studentOutput, trueLabel);
```

## FAQs

### Q: Why split into multiple classes instead of using enums?

**A**: The Open/Closed Principle states that code should be open for extension but closed for modification. With enums, adding a new strategy requires modifying the existing class (adding enum value + switch case). With the new architecture, you just create a new class that implements the interface.

### Q: Can I still use the old AdaptiveDistillationStrategy?

**A**: No, it has been removed in favor of specific strategy classes. This encourages better architecture and makes extension easier.

### Q: How do I choose between Confidence, Accuracy, and Entropy strategies?

**A**:
- **ConfidenceBasedAdaptiveStrategy**: Best default choice, works without labels
- **AccuracyBasedAdaptiveStrategy**: Use when you have labeled data and want to track correctness
- **EntropyBasedAdaptiveStrategy**: Use when you want a more holistic uncertainty measure

### Q: Can I create my own adaptive strategy?

**A**: Yes! Extend `AdaptiveDistillationStrategyBase<T>` and override `ComputeAdaptiveTemperature`. See "Creating Custom Strategies" section above.

## Summary

The refactoring:
- ✅ Follows SOLID principles (especially Open/Closed)
- ✅ Separates inference (teachers) from training (strategies)
- ✅ Makes extending with new strategies easy
- ✅ Improves testability through focused classes
- ✅ Enables composition and dependency injection
- ✅ Provides production-ready implementations

**Migration is straightforward**: Replace enum-based construction with specific strategy classes!
