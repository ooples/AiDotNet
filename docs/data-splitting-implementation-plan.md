# Data Splitting Implementation Plan

## Overview

This document outlines the implementation plan for 50+ data splitting methods as part of the DataPreparation pipeline in AiDotNet.

---

## Key Concepts: Data Preparation vs Data Preprocessing

### Data Preparation (Row-Changing Operations)
**What it does:** Changes the NUMBER of rows in your dataset
- **Outlier Removal:** Removes unusual data points (reduces rows)
- **Data Augmentation (SMOTE):** Creates synthetic samples (adds rows)
- **Data Splitting:** Divides data into train/validation/test sets

**When it happens:** ONLY during training (fit), never during prediction

**Why it's separate:** These operations fundamentally change dataset size and must keep X and y synchronized.

### Data Preprocessing (Transform Operations)
**What it does:** Transforms values WITHOUT changing row count
- **Scaling:** StandardScaler, MinMaxScaler, RobustScaler
- **Encoding:** OneHotEncoder, LabelEncoder
- **Imputation:** Fill missing values
- **Feature Engineering:** PolynomialFeatures

**When it happens:** During both training (fit_transform) and prediction (transform)

**Why it's separate:** These operations preserve dataset structure and can be applied to new data.

### The Pipeline Flow
```
Raw Data
    ↓
[Data Preparation] ← Outlier removal, Augmentation, Splitting (changes rows)
    ↓
Train/Val/Test Sets
    ↓
[Data Preprocessing] ← Scaling, Encoding (preserves rows)
    ↓
Ready for Model Training
```

---

## Architecture

### Interface: IDataSplitter<T>

```csharp
public interface IDataSplitter<T>
{
    /// <summary>
    /// Performs a single train/test split.
    /// </summary>
    DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null);

    /// <summary>
    /// Generates multiple splits (for cross-validation methods).
    /// </summary>
    IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null);

    /// <summary>
    /// Number of splits this splitter generates (1 for simple split, k for k-fold).
    /// </summary>
    int NumSplits { get; }

    /// <summary>
    /// Whether this splitter requires target labels (y).
    /// </summary>
    bool RequiresLabels { get; }

    /// <summary>
    /// Human-readable description of the splitting strategy.
    /// </summary>
    string Description { get; }
}
```

### Result Class: DataSplitResult<T>

```csharp
public class DataSplitResult<T>
{
    public Matrix<T> XTrain { get; init; }
    public Matrix<T> XTest { get; init; }
    public Vector<T>? yTrain { get; init; }
    public Vector<T>? yTest { get; init; }

    // Optional validation set (for three-way splits)
    public Matrix<T>? XValidation { get; init; }
    public Vector<T>? yValidation { get; init; }

    // Indices for tracking/debugging
    public int[] TrainIndices { get; init; }
    public int[] TestIndices { get; init; }
    public int[]? ValidationIndices { get; init; }

    // Fold information (for CV methods)
    public int? FoldIndex { get; init; }
    public int? TotalFolds { get; init; }
}
```

### Base Class: DataSplitterBase<T>

```csharp
public abstract class DataSplitterBase<T> : IDataSplitter<T>
{
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected readonly int _randomSeed;
    protected readonly Random _random;
    protected readonly bool _shuffle;

    protected DataSplitterBase(bool shuffle = true, int randomSeed = 42)
    {
        _shuffle = shuffle;
        _randomSeed = randomSeed;
        _random = RandomHelper.CreateSeededRandom(randomSeed);
    }

    // Common helpers
    protected int[] GetIndices(int count);
    protected void ShuffleIndices(int[] indices);
    protected void ValidateInputs(Matrix<T> X, Vector<T>? y);
    protected Matrix<T> SelectRows(Matrix<T> X, int[] indices);
    protected Vector<T> SelectElements(Vector<T> y, int[] indices);

    // Template methods
    public abstract DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null);
    public virtual IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        yield return Split(X, y);
    }

    public virtual int NumSplits => 1;
    public virtual bool RequiresLabels => false;
    public abstract string Description { get; }
}
```

---

## Integration with DataPreparationPipeline

### Updated Pipeline Interface

```csharp
public class DataPreparationPipeline<T>
{
    private readonly List<IRowOperation<T>> _operations;
    private IDataSplitter<T>? _splitter;

    // Existing methods
    public DataPreparationPipeline<T> Add(IRowOperation<T> operation);
    public DataPreparationPipeline<T> RemoveOutliers(IAnomalyDetector<T> detector, OutlierHandlingMode mode = OutlierHandlingMode.Remove);
    public DataPreparationPipeline<T> AddAugmentation(TabularAugmenterBase<T> augmenter);

    // New splitting methods
    public DataPreparationPipeline<T> WithSplitter(IDataSplitter<T> splitter);

    // Convenience methods with industry-standard defaults
    public DataPreparationPipeline<T> WithTrainTestSplit(double testSize = 0.2);
    public DataPreparationPipeline<T> WithTrainValTestSplit(double trainSize = 0.7, double valSize = 0.15);
    public DataPreparationPipeline<T> WithKFold(int k = 5);
    public DataPreparationPipeline<T> WithStratifiedKFold(int k = 5);
    public DataPreparationPipeline<T> WithTimeSeriesSplit(int nSplits = 5);

    // Execute pipeline
    public (Matrix<T> X, Vector<T> y) FitResample(Matrix<T> X, Vector<T> y);
    public DataSplitResult<T> FitResampleAndSplit(Matrix<T> X, Vector<T> y);
    public IEnumerable<DataSplitResult<T>> FitResampleAndGetSplits(Matrix<T> X, Vector<T> y);
}
```

### ConfigureDataPreparation Update

```csharp
// In AiModelBuilder
public IAiModelBuilder<T, TInput, TOutput> ConfigureDataPreparation(
    Action<DataPreparationPipeline<T>>? pipelineBuilder = null)
{
    _dataPreparationPipeline = new DataPreparationPipeline<T>();

    if (pipelineBuilder != null)
    {
        pipelineBuilder(_dataPreparationPipeline);
    }
    else
    {
        // Industry-standard defaults when user doesn't configure
        // - No outlier removal (user should explicitly choose)
        // - Train/Val/Test split with 70/15/15
        _dataPreparationPipeline.WithTrainValTestSplit();
    }

    DataPreparationRegistry<T>.Current = _dataPreparationPipeline;
    return this;
}
```

---

## Implementation: All 50+ Splitters

### Category 1: Basic Random Splitting (5 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 1 | `TrainTestSplitter<T>` | Simple train/test split | testSize=0.2 |
| 2 | `TrainValTestSplitter<T>` | Three-way split | trainSize=0.7, valSize=0.15 |
| 3 | `HoldoutSplitter<T>` | Multiple holdout test sets | testSize=0.2, numHoldouts=1 |
| 4 | `RandomSplitter<T>` | Alias for TrainTestSplitter | testSize=0.2 |
| 5 | `ShuffleSplitter<T>` | Repeated random splits (Monte Carlo) | nSplits=10, testSize=0.2 |

### Category 2: Cross-Validation (8 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 6 | `KFoldSplitter<T>` | K-fold cross-validation | k=5 |
| 7 | `StratifiedKFoldSplitter<T>` | K-fold preserving class distribution | k=5 |
| 8 | `RepeatedKFoldSplitter<T>` | Multiple k-fold runs | k=5, nRepeats=10 |
| 9 | `StratifiedRepeatedKFoldSplitter<T>` | Stratified repeated k-fold | k=5, nRepeats=10 |
| 10 | `LeaveOneOutSplitter<T>` | Each sample is test once | - |
| 11 | `LeavePOutSplitter<T>` | All combinations of p samples | p=2 |
| 12 | `StratifiedShuffleSplitter<T>` | Stratified Monte Carlo CV | nSplits=10, testSize=0.2 |
| 13 | `RepeatedStratifiedKFoldSplitter<T>` | Alias with clearer name | k=5, nRepeats=10 |

### Category 3: Time Series (8 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 14 | `TimeSeriesSplitter<T>` | Expanding window | nSplits=5 |
| 15 | `SlidingWindowSplitter<T>` | Fixed-size sliding window | windowSize, stepSize |
| 16 | `BlockedTimeSeriesSplitter<T>` | Gap between train/test | nSplits=5, gap=0 |
| 17 | `PurgedKFoldSplitter<T>` | Removes samples near test | k=5, purgeSize, embargoSize |
| 18 | `CombinatorialPurgedSplitter<T>` | All time period combinations | nGroups |
| 19 | `WalkForwardSplitter<T>` | Sequential train-test | initialTrainSize, stepSize |
| 20 | `AnchoredWalkForwardSplitter<T>` | Walk-forward fixed start | initialTrainSize, stepSize |
| 21 | `RollingOriginSplitter<T>` | Multi-step forecast eval | origin, horizon, step |

### Category 4: Group-Based (6 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 22 | `GroupKFoldSplitter<T>` | Keeps groups together | k=5 |
| 23 | `StratifiedGroupKFoldSplitter<T>` | Group k-fold + stratification | k=5 |
| 24 | `LeaveOneGroupOutSplitter<T>` | Each group is test once | - |
| 25 | `LeavePGroupsOutSplitter<T>` | All combinations of p groups | p=2 |
| 26 | `GroupShuffleSplitter<T>` | Random group-based splits | nSplits=10, testSize=0.2 |
| 27 | `PredefinedSplitter<T>` | User-specified split | trainIndices, testIndices |

### Category 5: Stratified (5 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 28 | `StratifiedTrainTestSplitter<T>` | Maintains class proportions | testSize=0.2 |
| 29 | `StratifiedTrainValTestSplitter<T>` | Three-way stratified | trainSize=0.7, valSize=0.15 |
| 30 | `IterativeStratificationSplitter<T>` | Multi-label stratification | testSize=0.2 |
| 31 | `DistributionPreservingSplitter<T>` | For regression targets | testSize=0.2, nBins=10 |
| 32 | `BalancedSplitter<T>` | Equal class representation | testSize=0.2 |

### Category 6: Cluster-Based (3 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 33 | `ClusterBasedSplitter<T>` | Split by clusters | nClusters, testSize=0.2 |
| 34 | `AntiClusteringSplitter<T>` | Maximizes diversity | testSize=0.2 |
| 35 | `SimilarityBasedSplitter<T>` | Based on sample similarity | testSize=0.2, threshold |

### Category 7: Adversarial/Robust (3 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 36 | `AdversarialValidationSplitter<T>` | Removes easy-to-distinguish | testSize=0.2, threshold=0.5 |
| 37 | `CovariateShiftSplitter<T>` | Creates distribution shift | shiftStrength |
| 38 | `OutOfDistributionSplitter<T>` | Test outside train dist | - |

### Category 8: Nested/Hierarchical (3 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 39 | `NestedCVSplitter<T>` | Inner + outer CV | outerFolds=5, innerFolds=3 |
| 40 | `DoubleCVSplitter<T>` | Two-level CV | outerFolds=5, innerFolds=5 |
| 41 | `HierarchicalSplitter<T>` | Multi-level data | levels[] |

### Category 9: Bootstrap-Based (4 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 42 | `BootstrapSplitter<T>` | Sample with replacement | nIterations=100 |
| 43 | `Bootstrap632Splitter<T>` | .632 bootstrap | nIterations=100 |
| 44 | `Bootstrap632PlusSplitter<T>` | .632+ bootstrap | nIterations=100 |
| 45 | `StratifiedBootstrapSplitter<T>` | Stratified bootstrap | nIterations=100 |

### Category 10: Domain-Specific (6 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 46 | `SpatialSplitter<T>` | Geographic splitting | blockSize or coordinates |
| 47 | `TemporalSpatialSplitter<T>` | Time + location | timeWeight, spaceWeight |
| 48 | `GraphSplitter<T>` | For graph data | splitType (node/edge) |
| 49 | `SequenceSplitter<T>` | For sequential data | sequenceColumn |
| 50 | `ImagePatchSplitter<T>` | Image segmentation | patchSize, overlap=0 |
| 51 | `MultiTaskSplitter<T>` | Consistent multi-task | taskColumn |

### Category 11: Active Learning (2 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 52 | `PoolBasedSplitter<T>` | Large unlabeled pool | initialLabeledSize |
| 53 | `QueryByCommitteeSplitter<T>` | Committee disagreement | committeeSize |

### Category 12: Federated Learning (3 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 54 | `IIDClientSplitter<T>` | Random client distribution | nClients |
| 55 | `NonIIDClientSplitter<T>` | Biased client distribution | nClients, skewType |
| 56 | `DirichletSplitter<T>` | Dirichlet-based non-IID | nClients, alpha=0.5 |

### Category 13: Incremental/Online (3 classes)

| # | Class Name | Description | Parameters |
|---|------------|-------------|------------|
| 57 | `PrequentialSplitter<T>` | Test-then-train | - |
| 58 | `LandmarkWindowSplitter<T>` | Fixed start, growing | landmarkIndex |
| 59 | `OnlineSlidingWindowSplitter<T>` | Stream mining | windowSize |

---

## Industry Standard Defaults

When user doesn't specify a splitter, auto-detect based on:

```csharp
private IDataSplitter<T> GetDefaultSplitter(Matrix<T> X, Vector<T>? y)
{
    int nSamples = X.Rows;
    bool isClassification = y != null && IsClassificationTarget(y);
    bool isTimeSeries = _options?.IsTimeSeries ?? false;

    if (isTimeSeries)
    {
        // Time series: no shuffling, temporal split
        return new TimeSeriesSplitter<T>(nSplits: 5);
    }

    if (nSamples < 100)
    {
        // Very small: Leave-One-Out
        return new LeaveOneOutSplitter<T>();
    }

    if (nSamples < 1000)
    {
        // Small: K-Fold CV (stratified if classification)
        return isClassification
            ? new StratifiedKFoldSplitter<T>(k: 10)
            : new KFoldSplitter<T>(k: 10);
    }

    if (nSamples < 10000)
    {
        // Medium: 5-Fold CV or 70/15/15 split
        return isClassification
            ? new StratifiedKFoldSplitter<T>(k: 5)
            : new KFoldSplitter<T>(k: 5);
    }

    // Large: Simple train/val/test split (stratified if classification)
    return isClassification
        ? new StratifiedTrainValTestSplitter<T>(trainSize: 0.7, valSize: 0.15)
        : new TrainValTestSplitter<T>(trainSize: 0.7, valSize: 0.15);
}
```

---

## File Structure

```
src/Preprocessing/DataPreparation/
├── IDataSplitter.cs                    # Interface
├── DataSplitResult.cs                  # Result class
├── DataSplitterBase.cs                 # Base class
├── Splitting/
│   ├── Basic/
│   │   ├── TrainTestSplitter.cs
│   │   ├── TrainValTestSplitter.cs
│   │   ├── HoldoutSplitter.cs
│   │   ├── RandomSplitter.cs
│   │   └── ShuffleSplitter.cs
│   ├── CrossValidation/
│   │   ├── KFoldSplitter.cs
│   │   ├── StratifiedKFoldSplitter.cs
│   │   ├── RepeatedKFoldSplitter.cs
│   │   ├── LeaveOneOutSplitter.cs
│   │   ├── LeavePOutSplitter.cs
│   │   └── ...
│   ├── TimeSeries/
│   │   ├── TimeSeriesSplitter.cs
│   │   ├── SlidingWindowSplitter.cs
│   │   ├── PurgedKFoldSplitter.cs
│   │   └── ...
│   ├── GroupBased/
│   │   ├── GroupKFoldSplitter.cs
│   │   ├── LeaveOneGroupOutSplitter.cs
│   │   └── ...
│   ├── Stratified/
│   │   ├── StratifiedTrainTestSplitter.cs
│   │   ├── IterativeStratificationSplitter.cs
│   │   └── ...
│   ├── Bootstrap/
│   │   ├── BootstrapSplitter.cs
│   │   ├── Bootstrap632Splitter.cs
│   │   └── ...
│   └── Specialized/
│       ├── SpatialSplitter.cs
│       ├── AdversarialValidationSplitter.cs
│       ├── FederatedSplitters.cs
│       └── ...
```

---

## Implementation Order

### Phase 1: Core Infrastructure
1. `IDataSplitter<T>` interface
2. `DataSplitResult<T>` result class
3. `DataSplitterBase<T>` base class
4. Update `DataPreparationPipeline<T>` to include splitter

### Phase 2: Basic Splitters (Most Common)
5. `TrainTestSplitter<T>`
6. `TrainValTestSplitter<T>`
7. `KFoldSplitter<T>`
8. `StratifiedKFoldSplitter<T>`
9. `StratifiedTrainTestSplitter<T>`

### Phase 3: Cross-Validation
10. `RepeatedKFoldSplitter<T>`
11. `LeaveOneOutSplitter<T>`
12. `LeavePOutSplitter<T>`
13. `ShuffleSplitter<T>`
14. `StratifiedShuffleSplitter<T>`

### Phase 4: Time Series
15. `TimeSeriesSplitter<T>`
16. `SlidingWindowSplitter<T>`
17. `BlockedTimeSeriesSplitter<T>`
18. `WalkForwardSplitter<T>`
19. `PurgedKFoldSplitter<T>`

### Phase 5: Group-Based
20. `GroupKFoldSplitter<T>`
21. `LeaveOneGroupOutSplitter<T>`
22. `GroupShuffleSplitter<T>`
23. `PredefinedSplitter<T>`

### Phase 6: Bootstrap & Advanced
24-59. Remaining splitters

---

## Delete/Replace

### Files to Delete
- `src/Preprocessing/TrainTestSplit.cs` (replaced by new splitters)
- `src/Preprocessing/DataPreparation/DataSplitter.cs` (if exists, my earlier creation)
- `src/Interfaces/IOutlierRemoval.cs`
- `src/AnomalyDetection/OutlierRemovalAdapter.cs`
- `src/AnomalyDetection/NoOutlierRemoval.cs`

### Files to Update
- `src/Preprocessing/DataPreparation/DataPreparationPipeline.cs` - add splitter support
- `src/AiModelBuilder.cs` - use new splitter system
- `src/Regression/GeneticAlgorithmRegression.cs` - remove legacy dependencies
- `src/Regression/SymbolicRegression.cs` - remove legacy dependencies
- `src/DataProcessor/DefaultDataPreprocessor.cs` - remove or deprecate

---

## Verification

1. Build: `dotnet build --no-restore`
2. Run existing tests
3. Add tests for each splitter category
4. Verify AiModelBuilder integration works
5. Test industry-standard defaults
