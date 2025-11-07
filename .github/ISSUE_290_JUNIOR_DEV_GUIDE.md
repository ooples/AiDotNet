# Issue #290: Junior Developer Implementation Guide

## Episodic Data Abstractions for Meta-Learning (N-way K-shot)

**This issue implements the data infrastructure for meta-learning - ALREADY COMPLETED!**

### What Was Built:

1. **MetaLearningTask<T, TInput, TOutput>**: Container for support and query sets
2. **IEpisodicDataLoader<T, TInput, TOutput>**: Interface for task sampling
3. **UniformEpisodicDataLoader**: Random N-way K-shot task sampling
4. **BalancedEpisodicDataLoader**: Balanced class distribution
5. **StratifiedEpisodicDataLoader**: Stratified sampling
6. **CurriculumEpisodicDataLoader**: Progressive difficulty tasks

**Status**: ✅ COMPLETED - All code implemented and tested

---

## Understanding Episodic Data Loading

### Why Meta-Learning Needs Special Data Loading

Traditional machine learning:
```
Data Loader:
- Samples: Individual examples
- Batches: (batch_size, features)
- Labels: (batch_size, classes)

Example: 32 images in a batch
```

Meta-learning:
```
Episodic Data Loader:
- Samples: Entire tasks (episodes)
- Support Set: (N*K, features) - for adaptation
- Query Set: (N*Q, features) - for evaluation

Example: 1 task with 25 support + 75 query = 100 images
```

**Key Difference**: Meta-learning samples **tasks**, not individual examples!

### What is N-way K-shot?

**N-way**: Number of classes per task
**K-shot**: Number of support examples per class
**Q-shot**: Number of query examples per class (typically 10-15)

**Example: 5-way 3-shot classification**

```
Task Structure:
├─ Support Set (training for this task)
│  ├─ Class 1 (cat): [img1, img2, img3]        (3 shots)
│  ├─ Class 2 (dog): [img4, img5, img6]        (3 shots)
│  ├─ Class 3 (bird): [img7, img8, img9]       (3 shots)
│  ├─ Class 4 (car): [img10, img11, img12]     (3 shots)
│  └─ Class 5 (tree): [img13, img14, img15]    (3 shots)
│  Total: 5 classes × 3 shots = 15 examples
│
└─ Query Set (testing for this task)
   ├─ Class 1 (cat): [img16-25]                (10 queries)
   ├─ Class 2 (dog): [img26-35]                (10 queries)
   ├─ Class 3 (bird): [img36-45]               (10 queries)
   ├─ Class 4 (car): [img46-55]                (10 queries)
   └─ Class 5 (tree): [img56-65]               (10 queries)
   Total: 5 classes × 10 queries = 50 examples
```

**Common Configurations**:

| Configuration | Support Size | Query Size | Use Case |
|--------------|--------------|------------|----------|
| 5-way 1-shot | 5 examples | 50 examples | Extreme few-shot |
| 5-way 5-shot | 25 examples | 75 examples | Standard benchmark |
| 10-way 5-shot | 50 examples | 150 examples | Many-class few-shot |
| 20-way 1-shot | 20 examples | 200 examples | Large-scale few-shot |

---

## Implemented Components

### 1. MetaLearningTask Data Structure

**File**: `src/Data/Abstractions/MetaLearningTask.cs`

**Purpose**: Container for one meta-learning task (episode).

**Structure**:
```csharp
public class MetaLearningTask<T, TInput, TOutput>
{
    // Support set (for adaptation)
    public TInput SupportSetX { get; set; }  // Input features
    public TOutput SupportSetY { get; set; } // Target labels

    // Query set (for evaluation)
    public TInput QuerySetX { get; set; }    // Input features
    public TOutput QuerySetY { get; set; }   // Target labels
}
```

**Example Usage**:
```csharp
// 5-way 5-shot task
var task = new MetaLearningTask<double, Tensor<double>, Tensor<double>>
{
    SupportSetX = new Tensor<double>(new[] { 25, 784 }),  // 25 images
    SupportSetY = new Tensor<double>(new[] { 25, 5 }),    // 25 labels (5 classes)
    QuerySetX = new Tensor<double>(new[] { 75, 784 }),    // 75 images
    QuerySetY = new Tensor<double>(new[] { 75, 5 })       // 75 labels
};

// Inner loop: Train on support set
model.Train(task.SupportSetX, task.SupportSetY);

// Evaluation: Test on query set
double loss = model.Evaluate(task.QuerySetX, task.QuerySetY);
```

**Key Concepts**:
- **Support Set**: Small labeled dataset for quick adaptation (like training data)
- **Query Set**: Held-out examples for evaluation (like test data)
- **No overlap**: Support and query sets MUST be disjoint (no shared examples)

### 2. IEpisodicDataLoader Interface

**File**: `src/Interfaces/IEpisodicDataLoader.cs`

**Purpose**: Define contract for task sampling.

**Interface**:
```csharp
public interface IEpisodicDataLoader<T, TInput, TOutput>
{
    /// <summary>
    /// Samples and returns the next N-way K-shot meta-learning task.
    /// </summary>
    MetaLearningTask<T, TInput, TOutput> GetNextTask();
}
```

**Why an interface?**
- Allows multiple sampling strategies (uniform, balanced, curriculum)
- Enables dependency injection
- Facilitates testing with mock data loaders
- Supports custom implementations

**Example Usage**:
```csharp
IEpisodicDataLoader<double, Tensor<double>, Tensor<double>> dataLoader =
    new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(...);

// Sample tasks for meta-training
for (int iter = 0; iter < 1000; iter++)
{
    var task = dataLoader.GetNextTask();
    // Meta-learning algorithm processes this task
}
```

### 3. UniformEpisodicDataLoader

**File**: `src/Data/Loaders/UniformEpisodicDataLoader.cs`

**Purpose**: Random N-way K-shot task sampling (most common).

**Algorithm**:
```
1. Sample N random classes from dataset
2. For each class:
   a. Get all examples of that class
   b. Shuffle examples
   c. Take first K for support set
   d. Take next Q for query set
3. Combine into MetaLearningTask
```

**Configuration**:
```csharp
var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
    datasetX: trainingFeatures,      // All training data
    datasetY: trainingLabels,         // All training labels
    nWay: 5,                          // 5 classes per task
    kShot: 5,                         // 5 support examples per class
    queryShots: 15                    // 15 query examples per class
);
```

**Implementation Details**:
```csharp
public MetaLearningTask<T, TInput, TOutput> GetNextTask()
{
    // Step 1: Sample N random classes
    var allClasses = _classIndices.Keys.ToList();
    var selectedClasses = RandomSample(allClasses, _nWay);

    var supportX = new List<TInput>();
    var supportY = new List<TOutput>();
    var queryX = new List<TInput>();
    var queryY = new List<TOutput>();

    // Step 2: For each selected class
    foreach (var classLabel in selectedClasses)
    {
        // Get all examples of this class
        var classExamples = _classIndices[classLabel];

        // Shuffle and split
        Shuffle(classExamples);
        var supportIndices = classExamples.Take(_kShot);
        var queryIndices = classExamples.Skip(_kShot).Take(_queryShots);

        // Add to support set
        foreach (var idx in supportIndices)
        {
            supportX.Add(_datasetX[idx]);
            supportY.Add(_datasetY[idx]);
        }

        // Add to query set
        foreach (var idx in queryIndices)
        {
            queryX.Add(_datasetX[idx]);
            queryY.Add(_datasetY[idx]);
        }
    }

    // Step 3: Create task
    return new MetaLearningTask<T, TInput, TOutput>
    {
        SupportSetX = Combine(supportX),
        SupportSetY = Combine(supportY),
        QuerySetX = Combine(queryX),
        QuerySetY = Combine(queryY)
    };
}
```

**Use When**:
- Standard meta-learning benchmarks
- All classes have sufficient examples
- No class imbalance concerns

### 4. BalancedEpisodicDataLoader

**File**: `src/Data/Loaders/BalancedEpisodicDataLoader.cs`

**Purpose**: Ensure balanced class sampling (all classes sampled equally often).

**Difference from Uniform**:
```
Uniform:
- Random class selection
- Some classes sampled more often
- May not see rare classes

Balanced:
- Tracks class sampling frequency
- Prioritizes under-sampled classes
- Ensures all classes seen equally
```

**Use When**:
- Dataset has class imbalance
- Want fair representation of all classes
- Long meta-training runs

### 5. StratifiedEpisodicDataLoader

**File**: `src/Data/Loaders/StratifiedEpisodicDataLoader.cs`

**Purpose**: Stratified sampling to maintain class distribution.

**Use When**:
- Want to preserve dataset's class distribution
- Statistical validity is important
- Evaluating on test set

### 6. CurriculumEpisodicDataLoader

**File**: `src/Data/Loaders/CurriculumEpisodicDataLoader.cs`

**Purpose**: Progressive difficulty (easy → hard tasks).

**Curriculum Strategy**:
```
Early training (iterations 0-300):
- 3-way 5-shot (easier: fewer classes)
- High visual similarity within classes
- Low inter-class variation

Mid training (iterations 300-700):
- 5-way 5-shot (medium difficulty)
- Moderate similarity

Late training (iterations 700-1000):
- 5-way 3-shot (harder: fewer shots)
- OR 10-way 5-shot (harder: more classes)
- High inter-class similarity (confusing classes)
```

**Use When**:
- Training is unstable
- Want faster convergence
- Model struggles with hard tasks early

---

## Understanding the Implementation

### Key Design Patterns

#### 1. Class Index Preprocessing

**Why preprocess?**
```csharp
// Naive approach (slow): Search entire dataset for each task
foreach (var example in allExamples)
{
    if (example.Label == targetClass)
        selectedExamples.Add(example);
}

// Optimized approach (fast): Pre-build index
Dictionary<int, List<int>> _classIndices;
// _classIndices[3] = [42, 78, 91, ...] (all indices where class = 3)

// Now sampling is O(1) instead of O(N)
var class3Examples = _classIndices[3];
```

**Implementation**:
```csharp
private void BuildClassIndex()
{
    _classIndices = new Dictionary<int, List<int>>();

    for (int i = 0; i < _datasetSize; i++)
    {
        int classLabel = GetClassLabel(_datasetY[i]);

        if (!_classIndices.ContainsKey(classLabel))
            _classIndices[classLabel] = new List<int>();

        _classIndices[classLabel].Add(i);
    }
}
```

#### 2. Support/Query Split

**Critical Requirement**: NO overlap between support and query sets!

```csharp
// WRONG: Could sample same example twice
var supportIndices = RandomSample(classExamples, kShot);
var queryIndices = RandomSample(classExamples, queryShots);

// CORRECT: Sequential split (no overlap)
Shuffle(classExamples);
var supportIndices = classExamples.Take(kShot);
var queryIndices = classExamples.Skip(kShot).Take(queryShots);
```

**Why no overlap?**
- Support set = training data
- Query set = test data
- Overlap = data leakage (model memorizes instead of generalizes)

#### 3. Generic Type Handling

**Supporting Multiple Data Formats**:

```csharp
// Works with Tensor<T>
var dataLoader1 = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(...);

// Works with Matrix<T>
var dataLoader2 = new UniformEpisodicDataLoader<float, Matrix<float>, Vector<float>>(...);

// Works with arrays
var dataLoader3 = new UniformEpisodicDataLoader<double, double[], double[]>(...);
```

**Implementation uses**:
- `TInput` for flexibility (any input format)
- `TOutput` for flexibility (any output format)
- `T` for numeric operations (any numeric type)

---

## Common Use Cases

### 1. Image Classification (Omniglot, Mini-ImageNet)

```csharp
// Load image dataset
var (images, labels) = LoadOmniglotDataset();
// images: (N, 28, 28, 1) - grayscale images
// labels: (N, num_classes) - one-hot encoded

var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
    datasetX: images,
    datasetY: labels,
    nWay: 5,          // 5 character classes per task
    kShot: 1,         // 1-shot learning (very hard!)
    queryShots: 15    // 15 query examples per class
);

// Each task: Learn to recognize 5 new characters from 1 example each!
```

### 2. Regression (Sine Wave Fitting)

```csharp
// Generate sine wave dataset
var (waveInputs, waveOutputs) = GenerateSineWaves(numTasks: 1000);
// waveInputs: (N, 1) - x values
// waveOutputs: (N, 1) - y = A*sin(x + φ)

var dataLoader = new UniformEpisodicDataLoader<double, Matrix<double>, Vector<double>>(
    datasetX: waveInputs,
    datasetY: waveOutputs,
    nWay: 1,          // Regression: 1 "class" (the function)
    kShot: 10,        // 10 points to fit from
    queryShots: 50    // 50 points to evaluate on
);

// Each task: Fit a sine wave from 10 (x,y) points!
```

### 3. Text Classification (Few-Shot Intent Detection)

```csharp
// Load text embeddings
var (textEmbeddings, intents) = LoadIntentDataset();
// textEmbeddings: (N, 768) - BERT embeddings
// intents: (N, num_intents) - intent labels

var dataLoader = new UniformEpisodicDataLoader<float, Matrix<float>, Vector<float>>(
    datasetX: textEmbeddings,
    datasetY: intents,
    nWay: 5,          // 5 intents per task
    kShot: 5,         // 5 example utterances per intent
    queryShots: 20    // 20 test utterances
);

// Each task: Classify new user intents from 5 examples each!
```

---

## Testing the Implementation

### Unit Tests

**File**: `tests/UnitTests/Data/UniformEpisodicDataLoaderTests.cs`

**Test 1: Task Dimensions**
```csharp
[Fact]
public void GetNextTask_Returns_CorrectDimensions()
{
    // Arrange: 5-way 3-shot, 10-query
    var dataLoader = CreateTestLoader(nWay: 5, kShot: 3, queryShots: 10);

    // Act
    var task = dataLoader.GetNextTask();

    // Assert
    Assert.Equal(15, task.SupportSetX.Shape[0]);  // 5 classes × 3 shots
    Assert.Equal(50, task.QuerySetX.Shape[0]);    // 5 classes × 10 queries
}
```

**Test 2: No Overlap**
```csharp
[Fact]
public void GetNextTask_NoOverlap_BetweenSupportAndQuery()
{
    // Arrange
    var dataLoader = CreateTestLoaderWithUniqueIds();

    // Act
    var task = dataLoader.GetNextTask();

    // Get unique IDs from support and query
    var supportIds = ExtractIds(task.SupportSetX);
    var queryIds = ExtractIds(task.QuerySetX);

    // Assert: No intersection
    Assert.Empty(supportIds.Intersect(queryIds));
}
```

**Test 3: Class Distribution**
```csharp
[Fact]
public void GetNextTask_CorrectClassDistribution()
{
    // Arrange: 5-way 5-shot
    var dataLoader = CreateTestLoader(nWay: 5, kShot: 5, queryShots: 15);

    // Act
    var task = dataLoader.GetNextTask();

    // Get unique classes in support set
    var uniqueClasses = GetUniqueClasses(task.SupportSetY);

    // Assert: Exactly 5 unique classes
    Assert.Equal(5, uniqueClasses.Count);

    // Assert: Each class has exactly 5 support examples
    foreach (var classLabel in uniqueClasses)
    {
        int count = CountExamplesOfClass(task.SupportSetY, classLabel);
        Assert.Equal(5, count);
    }
}
```

---

## Common Pitfalls to Avoid

### 1. Data Leakage (Overlapping Support and Query)

❌ **WRONG**:
```csharp
// Could sample same example in both sets
var allIndices = GetAllIndicesForClass(classLabel);
var supportIndices = RandomSample(allIndices, kShot);
var queryIndices = RandomSample(allIndices, queryShots);
```

✅ **CORRECT**:
```csharp
// Sequential split ensures no overlap
var allIndices = GetAllIndicesForClass(classLabel);
Shuffle(allIndices);
var supportIndices = allIndices.Take(kShot);
var queryIndices = allIndices.Skip(kShot).Take(queryShots);
```

### 2. Insufficient Examples Per Class

❌ **WRONG**:
```csharp
// Class has 10 examples, but need 5 + 15 = 20!
kShot = 5;
queryShots = 15;  // Crash!
```

✅ **CORRECT**:
```csharp
// Validate during construction
int requiredExamplesPerClass = kShot + queryShots;
foreach (var classExamples in _classIndices.Values)
{
    if (classExamples.Count < requiredExamplesPerClass)
        throw new ArgumentException(
            $"Class has {classExamples.Count} examples, need {requiredExamplesPerClass}");
}
```

### 3. Not Shuffling Before Split

❌ **WRONG**:
```csharp
// Always takes first K examples (not random)
var supportIndices = classExamples.Take(kShot);
```

✅ **CORRECT**:
```csharp
// Shuffle first for randomness
Shuffle(classExamples);
var supportIndices = classExamples.Take(kShot);
```

### 4. Forgetting to Clone/Copy Data

❌ **WRONG**:
```csharp
// All tasks share same reference!
task.SupportSetX = originalData;
```

✅ **CORRECT**:
```csharp
// Create new copies for each task
task.SupportSetX = CopyData(originalData, indices);
```

---

## Performance Considerations

### 1. Pre-build Class Index (Done in Constructor)

**Time Complexity**:
- Without index: O(N) per task (scan entire dataset)
- With index: O(N*K) per task (only copy selected examples)

**Space Complexity**:
- Index: O(N) extra memory (list of integers)
- Worth it for faster sampling!

### 2. Batch Task Generation

```csharp
// Generate multiple tasks at once for efficiency
public List<MetaLearningTask<T, TInput, TOutput>> GetTaskBatch(int batchSize)
{
    var tasks = new List<MetaLearningTask<T, TInput, TOutput>>();
    for (int i = 0; i < batchSize; i++)
    {
        tasks.Add(GetNextTask());
    }
    return tasks;
}
```

### 3. Caching vs On-Demand

**Current implementation**: On-demand generation
- **Pros**: Low memory usage, infinite task variations
- **Cons**: CPU overhead per task

**Alternative**: Pre-generate task cache
- **Pros**: Faster training, no sampling overhead
- **Cons**: High memory usage, limited task diversity

**Recommendation**: Use on-demand for most cases.

---

## Integration with Meta-Learners

### Usage in Reptile

```csharp
var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(...);

var metaLearner = new ReptileTrainer<double, Tensor<double>, Tensor<double>>(
    metaModel: model,
    lossFunction: loss,
    dataLoader: dataLoader,  // Configured at construction
    config: config
);

// Data loader is used internally
metaLearner.Train();  // Automatically samples tasks via dataLoader
```

### Usage in MAML/SEAL (Same Pattern)

```csharp
var metaLearner = new MAMLTrainer<double, Tensor<double>, Tensor<double>>(
    metaModel: model,
    lossFunction: loss,
    dataLoader: dataLoader,  // Same interface!
    config: config
);
```

---

## Summary

### What Was Accomplished

✅ **MetaLearningTask**: Data structure for episodic tasks
✅ **IEpisodicDataLoader**: Interface for task sampling
✅ **UniformEpisodicDataLoader**: Random N-way K-shot sampling
✅ **BalancedEpisodicDataLoader**: Balanced class sampling
✅ **StratifiedEpisodicDataLoader**: Stratified sampling
✅ **CurriculumEpisodicDataLoader**: Progressive difficulty
✅ **Comprehensive tests**: Unit and integration tests
✅ **Full documentation**: XML docs with beginner explanations

### Key Takeaways

1. **Episodic data loading is fundamental to meta-learning**
   - Samples tasks, not individual examples
   - Each task has support (train) and query (test) sets

2. **N-way K-shot defines task structure**
   - N = number of classes
   - K = support examples per class
   - Q = query examples per class

3. **Multiple sampling strategies available**
   - Uniform: Random sampling (most common)
   - Balanced: Equal class representation
   - Stratified: Preserve distribution
   - Curriculum: Progressive difficulty

4. **Proper implementation is critical**
   - No support/query overlap (prevents data leakage)
   - Pre-build class index (performance)
   - Support multiple data formats (generics)
   - Validate sufficient examples per class

### Dependencies

This issue is a **dependency** for:
- ✅ Issue #292: Reptile (uses episodic data loader)
- ⏳ Issue #291: MAML (uses episodic data loader)
- ⏳ Issue #289: SEAL (uses episodic data loader)
- ⏳ Issue #288: Documentation and examples

All meta-learning algorithms depend on episodic data loading!

---

## Further Reading

### Implemented Files
- `src/Data/Abstractions/MetaLearningTask.cs`
- `src/Interfaces/IEpisodicDataLoader.cs`
- `src/Data/Loaders/EpisodicDataLoaderBase.cs`
- `src/Data/Loaders/UniformEpisodicDataLoader.cs`
- `src/Data/Loaders/BalancedEpisodicDataLoader.cs`
- `src/Data/Loaders/StratifiedEpisodicDataLoader.cs`
- `src/Data/Loaders/CurriculumEpisodicDataLoader.cs`

### Test Files
- `tests/UnitTests/Data/UniformEpisodicDataLoaderTests.cs`
- `tests/UnitTests/Data/AdvancedEpisodicDataLoaderTests.cs`

### Related Issues
- Issue #286: Meta-Learning Suite (master epic)
- Issue #292: Reptile (uses this infrastructure)
- Issue #291: MAML (uses this infrastructure)
- Issue #289: SEAL (uses this infrastructure)
