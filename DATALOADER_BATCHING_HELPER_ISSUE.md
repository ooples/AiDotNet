# DataLoader and Batching Utilities for Training Optimization

## User Story

> As a machine learning developer implementing or using optimizers, I want a standardized, reusable way to iterate over training data in batches with shuffling and splitting, so that I don't have to write repetitive batching boilerplate in every optimizer and can focus on the actual optimization algorithm.

---

## Problem Statement

**Current State:**
The codebase has **40+ repetitive instances** of manual batching code across optimizers, cross-validators, and other training components:

```csharp
// This pattern is repeated in 10+ optimizer classes:
for (int epoch = 0; epoch < maxEpochs; epoch++)
{
    var shuffledIndices = ShuffleIndices(indices, random);
    int numBatches = (int)Math.Ceiling((double)batchSize / batchSize);

    for (int i = 0; i < numBatches; i++)
    {
        var batchIndices = shuffledIndices.Skip(i * batchSize).Take(batchSize).ToArray();
        var xBatch = InputHelper<T, TInput>.GetBatch(xTrain, batchIndices);
        var yBatch = InputHelper<T, TOutput>.GetBatch(yTrain, batchIndices);

        // ... actual optimizer logic
    }
}
```

**Problems with Current Approach:**
1. **Code Duplication:** Batching logic copied 40+ times across codebase
2. **Inconsistency:** Different optimizers implement shuffling differently
3. **Error-Prone:** Off-by-one errors, incomplete final batches handled inconsistently
4. **Hard to Test:** Batching logic embedded in optimizer implementations
5. **Poor Separation of Concerns:** Optimizers responsible for data iteration AND optimization
6. **Difficult to Extend:** Adding new batching strategies (stratified, weighted) requires changes to multiple files

**Missing Features:**
- No stratified batching (maintain class distribution in each batch)
- No weighted sampling (for imbalanced datasets)
- No automatic drop_last handling for incomplete batches
- No consistent shuffle seeding across optimizers
- No batch prefetching or parallel loading

---

## Proposed Solution

Create a **DataLoader** utility that handles all data iteration, batching, and shuffling concerns, separating them from optimizer logic.

### Design Philosophy

1. **Separation of Concerns:** DataLoader handles data iteration, optimizers handle optimization
2. **Lazy Evaluation:** Use `IEnumerable<T>` and `yield return` for memory efficiency
3. **Compatibility:** Works with existing `InputHelper<T, TInput>` infrastructure
4. **Extensibility:** Easy to add new batching strategies via inheritance
5. **Testability:** Batching logic isolated and independently testable

---

## Architecture

### Phase 1: Core DataLoader (Foundation)

**Goal:** Replace all repetitive batching code with a single, reusable `DataLoader<T, TInput, TOutput>` class.

#### AC 1.1: Create `DataLoader<T, TInput, TOutput>` class (8 points)

**Requirements:**
- [ ] Create `src/Data/Batching/DataLoader.cs`
- [ ] Define `public class DataLoader<T, TInput, TOutput>`
- [ ] Constructor signature:
  ```csharp
  public DataLoader(
      TInput X,
      TOutput y,
      int batchSize,
      bool shuffle = true,
      bool dropLast = false,
      int? randomSeed = null)
  ```
- [ ] Implement `IEnumerable<(TInput X, TOutput y)>` to support `foreach` iteration
- [ ] Use `InputHelper<T, TInput>.GetBatchSize(X)` to determine dataset size
- [ ] Use `InputHelper<T, TInput>.GetBatch(X, indices)` for actual batching
- [ ] Store `batchSize`, `shuffle`, `dropLast`, `randomSeed` as readonly fields

**Implementation Details:**
```csharp
public class DataLoader<T, TInput, TOutput>
{
    private readonly TInput _X;
    private readonly TOutput _y;
    private readonly int _batchSize;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly int? _randomSeed;
    private readonly int _numSamples;

    public DataLoader(
        TInput X,
        TOutput y,
        int batchSize,
        bool shuffle = true,
        bool dropLast = false,
        int? randomSeed = null)
    {
        if (batchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");

        _X = X ?? throw new ArgumentNullException(nameof(X));
        _y = y ?? throw new ArgumentNullException(nameof(y));
        _batchSize = batchSize;
        _shuffle = shuffle;
        _dropLast = dropLast;
        _randomSeed = randomSeed;

        _numSamples = InputHelper<T, TInput>.GetBatchSize(X);

        // Validate X and y have same number of samples
        int numLabels = InputHelper<T, TOutput>.GetBatchSize(y);
        if (_numSamples != numLabels)
            throw new ArgumentException($"X has {_numSamples} samples but y has {numLabels}.");
    }

    public IEnumerable<(TInput X, TOutput y)> GetBatches()
    {
        // Generate indices
        var indices = Enumerable.Range(0, _numSamples).ToArray();

        // Shuffle if requested
        if (_shuffle)
        {
            var random = _randomSeed.HasValue
                ? new Random(_randomSeed.Value)
                : new Random();
            ShuffleInPlace(indices, random);
        }

        // Compute number of batches
        int numBatches = _dropLast
            ? _numSamples / _batchSize
            : (int)Math.Ceiling((double)_numSamples / _batchSize);

        // Yield batches
        for (int i = 0; i < numBatches; i++)
        {
            int startIdx = i * _batchSize;
            int endIdx = Math.Min(startIdx + _batchSize, _numSamples);
            int currentBatchSize = endIdx - startIdx;

            // Create batch indices
            var batchIndices = new int[currentBatchSize];
            Array.Copy(indices, startIdx, batchIndices, 0, currentBatchSize);

            // Extract batch data using InputHelper
            var xBatch = InputHelper<T, TInput>.GetBatch(_X, batchIndices);
            var yBatch = InputHelper<T, TOutput>.GetBatch(_y, batchIndices);

            yield return (xBatch, yBatch);
        }
    }

    private static void ShuffleInPlace(int[] array, Random random)
    {
        // Fisher-Yates shuffle
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    // Properties for introspection
    public int BatchSize => _batchSize;
    public int NumSamples => _numSamples;
    public int NumBatches => _dropLast
        ? _numSamples / _batchSize
        : (int)Math.Ceiling((double)_numSamples / _batchSize);
}
```

**Validation:**
- [ ] Throws `ArgumentNullException` if `X` or `y` is null
- [ ] Throws `ArgumentOutOfRangeException` if `batchSize < 1`
- [ ] Throws `ArgumentException` if X and y have different number of samples
- [ ] Handles incomplete final batch correctly when `dropLast = false`
- [ ] Drops incomplete final batch when `dropLast = true`
- [ ] Produces deterministic shuffle when `randomSeed` is set
- [ ] Produces different shuffle on each call when `randomSeed` is null

#### AC 1.2: Create Extension Method for Convenient Usage (2 points)

**Requirements:**
- [ ] Create `src/Data/Batching/DataLoaderExtensions.cs`
- [ ] Add extension method to make usage more fluent:

```csharp
public static class DataLoaderExtensions
{
    /// <summary>
    /// Creates batches from input and output data.
    /// </summary>
    public static IEnumerable<(TInput X, TOutput y)> CreateBatches<T, TInput, TOutput>(
        this (TInput X, TOutput y) data,
        int batchSize,
        bool shuffle = true,
        bool dropLast = false,
        int? randomSeed = null)
    {
        var dataLoader = new DataLoader<T, TInput, TOutput>(
            data.X, data.y, batchSize, shuffle, dropLast, randomSeed);

        return dataLoader.GetBatches();
    }
}
```

**Usage Example:**
```csharp
// Before (in optimizer):
for (int epoch = 0; epoch < maxEpochs; epoch++)
{
    var shuffled = ShuffleIndices(indices, random);
    for (int i = 0; i < numBatches; i++)
    {
        var batchIndices = shuffled.Skip(i * batchSize).Take(batchSize).ToArray();
        var xBatch = InputHelper<T, TInput>.GetBatch(xTrain, batchIndices);
        var yBatch = InputHelper<T, TOutput>.GetBatch(yTrain, batchIndices);

        // Update weights...
    }
}

// After (in optimizer):
for (int epoch = 0; epoch < maxEpochs; epoch++)
{
    foreach (var (xBatch, yBatch) in (xTrain, yTrain).CreateBatches<T, TInput, TOutput>(
        batchSize, shuffle: true, randomSeed: epoch))
    {
        // Update weights...
    }
}
```

---

### Phase 2: Optimizer Integration (High Impact)

**Goal:** Replace manual batching code in all optimizers with DataLoader.

#### AC 2.1: Refactor MiniBatchGradientDescentOptimizer (5 points)

**Requirements:**
- [ ] Open `src/Optimizers/MiniBatchGradientDescentOptimizer.cs`
- [ ] Replace manual batching loop in `Optimize` method with `DataLoader`
- [ ] Remove custom shuffle logic
- [ ] Remove manual batch index calculation
- [ ] Verify tests still pass with identical behavior
- [ ] Benchmark to ensure no performance regression

**Before/After:**
```csharp
// BEFORE (current code):
var indices = Enumerable.Range(0, batchSize).ToArray();
for (int epoch = 0; epoch < MaxEpochs; epoch++)
{
    var shuffled = ShuffleIndices(indices);
    for (int i = 0; i < numBatches; i++)
    {
        var batchIndices = shuffled.Skip(i * BatchSize).Take(BatchSize).ToArray();
        var xBatch = InputHelper<T, TInput>.GetBatch(xTrain, batchIndices);
        var yBatch = InputHelper<T, TOutput>.GetBatch(yTrain, batchIndices);

        // ... gradient update
    }
}

// AFTER (with DataLoader):
for (int epoch = 0; epoch < MaxEpochs; epoch++)
{
    var dataLoader = new DataLoader<T, TInput, TOutput>(
        inputData.XTrain,
        inputData.YTrain,
        batchSize: BatchSize,
        shuffle: true,
        dropLast: false,
        randomSeed: epoch);

    foreach (var (xBatch, yBatch) in dataLoader.GetBatches())
    {
        // ... gradient update (unchanged)
    }
}
```

#### AC 2.2: Refactor Remaining Optimizers (13 points total)

**Requirements:** Refactor these optimizer classes to use DataLoader:
- [ ] `AdamOptimizer.cs` (3 points)
- [ ] `RMSPropOptimizer.cs` (2 points)
- [ ] `AdagradOptimizer.cs` (2 points)
- [ ] `AdadeltaOptimizer.cs` (2 points)
- [ ] `NadamOptimizer.cs` (2 points)
- [ ] `AdamaxOptimizer.cs` (2 points)

**For each optimizer:**
- [ ] Replace manual batching with DataLoader
- [ ] Ensure shuffle seed is deterministic per epoch
- [ ] Remove duplicate shuffle/batching code
- [ ] Verify tests pass
- [ ] Update XML documentation if needed

---

### Phase 3: Cross-Validator Integration (Medium Impact)

**Goal:** Use DataLoader in cross-validators to ensure consistent batching behavior.

#### AC 3.1: Refactor CrossValidatorBase (5 points)

**Requirements:**
- [ ] Open `src/CrossValidators/CrossValidatorBase.cs`
- [ ] In `ValidateFold` method, replace manual `InputHelper.GetBatch` calls with DataLoader
- [ ] Add option to CrossValidatorOptions for batch processing during validation
- [ ] Update all derived cross-validators to benefit from DataLoader

**Impact:**
- Ensures consistent batching across training and validation
- Enables future batch-level metrics (batch accuracy, batch loss)
- Simplifies cross-validator implementations

---

### Phase 4: Advanced Features (Optional Enhancements)

**Goal:** Add advanced batching strategies for specialized use cases.

#### AC 4.1: Stratified Batching (8 points)

**Requirements:**
- [ ] Create `StratifiedDataLoader<T, TInput, TOutput>` inheriting from `DataLoader`
- [ ] Override `GetBatches()` to ensure each batch maintains class distribution
- [ ] Use stratified sampling to create balanced batches
- [ ] Add tests for multiclass classification datasets

**Use Case:**
Imbalanced datasets where each batch should have similar class distribution.

#### AC 4.2: Weighted Sampling (8 points)

**Requirements:**
- [ ] Add `sampleWeights` parameter to DataLoader constructor
- [ ] Implement weighted random sampling when `sampleWeights` is provided
- [ ] Ensure weights are normalized to probabilities
- [ ] Add tests for weighted sampling behavior

**Use Case:**
Over-sample minority classes or under-sample majority classes.

#### AC 4.3: Train/Validation Split Helper (5 points)

**Requirements:**
- [ ] Create static method `DataLoader.Split<T, TInput, TOutput>(...)`:
  ```csharp
  public static (DataLoader<T, TInput, TOutput> train, DataLoader<T, TInput, TOutput> val) Split(
      TInput X,
      TOutput y,
      double trainRatio = 0.8,
      int batchSize = 32,
      bool shuffle = true,
      int? randomSeed = null)
  ```
- [ ] Automatically split data into train/validation sets
- [ ] Create separate DataLoaders for each split
- [ ] Add tests for split ratios

**Use Case:**
Quick train/val splitting without needing separate cross-validator.

---

## Testing Requirements

### Unit Tests (src/Tests/UnitTests/Data/DataLoaderTests.cs)

- [ ] **Test_Constructor_ValidatesInputs:** Null checks, batch size validation, X/y length mismatch
- [ ] **Test_GetBatches_NoBatching:** Single batch when `batchSize >= numSamples`
- [ ] **Test_GetBatches_EvenBatches:** Batches all same size when `numSamples % batchSize == 0`
- [ ] **Test_GetBatches_IncompleteFinalBatch_DropFalse:** Final batch smaller, all samples included
- [ ] **Test_GetBatches_IncompleteFinalBatch_DropTrue:** Final batch dropped, some samples excluded
- [ ] **Test_GetBatches_Shuffle_Deterministic:** Same seed produces same order across calls
- [ ] **Test_GetBatches_Shuffle_Random:** No seed produces different order across calls
- [ ] **Test_GetBatches_AllSamplesIncluded:** Sum of all batch sizes equals numSamples (when dropLast=false)
- [ ] **Test_NumBatches_Property:** Correctly calculates number of batches
- [ ] **Test_ExtensionMethod_CreateBatches:** Extension method works correctly

### Integration Tests (src/Tests/IntegrationTests/Optimizers/)

- [ ] **Test_MiniBatchGD_WithDataLoader:** Verify optimizer produces same results with DataLoader
- [ ] **Test_Adam_WithDataLoader:** Verify optimizer produces same results with DataLoader
- [ ] **Test_CrossValidator_WithDataLoader:** Verify cross-validation works with DataLoader

### Performance Tests

- [ ] **Benchmark_DataLoader_vs_Manual:** Ensure DataLoader is within 5% performance of manual approach
- [ ] **Benchmark_LargeDataset:** Test with 1M+ samples to ensure memory efficiency

---

## Migration Guide

### For Optimizer Developers

**Old Pattern:**
```csharp
var indices = Enumerable.Range(0, numSamples).ToArray();
for (int epoch = 0; epoch < epochs; epoch++)
{
    ShuffleArray(indices);
    for (int i = 0; i < numBatches; i++)
    {
        var batchIndices = indices.Skip(i * batchSize).Take(batchSize).ToArray();
        var xBatch = InputHelper<T, TInput>.GetBatch(X, batchIndices);
        var yBatch = InputHelper<T, TOutput>.GetBatch(y, batchIndices);
        // ... training logic
    }
}
```

**New Pattern:**
```csharp
for (int epoch = 0; epoch < epochs; epoch++)
{
    var dataLoader = new DataLoader<T, TInput, TOutput>(
        X, y, batchSize, shuffle: true, randomSeed: epoch);

    foreach (var (xBatch, yBatch) in dataLoader.GetBatches())
    {
        // ... training logic (unchanged)
    }
}
```

### For Library Users

**No Breaking Changes:**
- Optimizers automatically use DataLoader internally
- Public API remains unchanged
- Existing code continues to work

**Optional: Direct DataLoader Usage**
```csharp
// For custom training loops:
var dataLoader = new DataLoader<double, Matrix<double>, Vector<double>>(
    trainingData, trainingLabels,
    batchSize: 64,
    shuffle: true);

foreach (var (xBatch, yBatch) in dataLoader.GetBatches())
{
    // Custom training logic
}
```

---

## Definition of Done

- [ ] `DataLoader<T, TInput, TOutput>` class implemented with all required features
- [ ] Extension method `CreateBatches` implemented
- [ ] All optimizers refactored to use DataLoader
- [ ] All manual batching code removed from optimizers
- [ ] All unit tests pass with >= 90% code coverage
- [ ] Integration tests confirm identical behavior before/after refactoring
- [ ] Performance benchmarks show no regression (within 5%)
- [ ] XML documentation complete with beginner-friendly examples
- [ ] Migration guide written for optimizer developers
- [ ] Code review approved
- [ ] All architectural requirements from `.github/USER_STORY_ARCHITECTURAL_REQUIREMENTS.md` followed

---

## ⚠️ CRITICAL ARCHITECTURAL REQUIREMENTS

**Before implementing this user story, you MUST review:**
- **📋 Full Requirements:** [`.github/USER_STORY_ARCHITECTURAL_REQUIREMENTS.md`](../.github/USER_STORY_ARCHITECTURAL_REQUIREMENTS.md)
- **📐 Project Rules:** [`.github/PROJECT_RULES.md`](../.github/PROJECT_RULES.md)

### Mandatory Implementation Checklist

#### 1. Class Organization (REQUIRED)
- [ ] Create `src/Data/Batching/DataLoader.cs` (one class per file)
- [ ] Create `src/Data/Batching/DataLoaderExtensions.cs` (extension methods)
- [ ] Do NOT create interfaces in subfolders - put in `src/Interfaces/` if needed
- [ ] Namespace: `namespace AiDotNet.Data.Batching;`

#### 2. Property Initialization (CRITICAL)
- [ ] NEVER use `default!` operator
- [ ] Initialize all properties with appropriate defaults
- [ ] Use `= string.Empty` for strings, `= new List<T>()` for collections

#### 3. Documentation (REQUIRED)
- [ ] XML documentation for all public members
- [ ] `<summary>`, `<param>`, `<returns>`, `<exception>` tags
- [ ] `<b>For Beginners:</b>` section explaining what DataLoader does
- [ ] Example usage in remarks section
- [ ] Explain shuffle algorithm (Fisher-Yates)
- [ ] Document dropLast behavior with examples

#### 4. Testing (REQUIRED)
- [ ] Minimum 90% code coverage
- [ ] Test with multiple numeric types (double, float, int)
- [ ] Test edge cases (empty data, single sample, exact batch size)
- [ ] Test shuffle determinism with seeds
- [ ] Integration tests with actual optimizers

#### 5. Performance (REQUIRED)
- [ ] Use `yield return` for lazy evaluation
- [ ] Avoid unnecessary array allocations
- [ ] Benchmark against current manual approach
- [ ] Document any performance considerations

#### 6. Backward Compatibility (CRITICAL)
- [ ] Do NOT break existing optimizer APIs
- [ ] Refactor optimizers INTERNALLY only
- [ ] Ensure existing tests pass without modification
- [ ] No changes to public optimizer interfaces

---

## Success Metrics

**Code Quality:**
- Reduce batching code duplication from 40+ instances to 1 reusable class
- Achieve >= 90% test coverage for DataLoader
- Zero performance regression in optimizer benchmarks

**Developer Experience:**
- Simplify optimizer implementations (fewer lines of code)
- Consistent batching behavior across all optimizers
- Easier to add new optimizers (less boilerplate)

**User Experience:**
- No breaking changes to existing code
- Optional: Users can use DataLoader directly for custom training loops
- Better error messages for invalid batch sizes, mismatched data

---

## Related Issues

- #282 - Dataset and DataLoader Abstractions (this is complementary, not duplicate)
- #295 - IBatchProvider interface (to be closed as redundant per analysis)

---

## Labels

- `enhancement` - Improves existing functionality
- `refactoring` - Code quality improvement
- `optimizer` - Affects optimizer implementations
- `data-loading` - Data pipeline improvements

---

**Estimated Effort:** 21-34 story points across all phases
- Phase 1: 10 points (foundation)
- Phase 2: 18 points (optimizer refactoring)
- Phase 3: 5 points (cross-validator integration)
- Phase 4: 21 points (advanced features, optional)

**Priority:** High - Reduces technical debt and simplifies optimizer development

**Status:** Draft - Ready for Review
