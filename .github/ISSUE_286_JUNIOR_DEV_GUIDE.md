# Issue #286: Junior Developer Implementation Guide

## Meta-Learning Suite (MAML/Reptile/iMAML/ALFA) and SEAL Investigation

**This is the master epic for tracking the entire Meta-Learning Suite implementation.**

### What You're Building:

This is an **epic** (a collection of related features), NOT a single implementation task. It coordinates four major sub-issues:

1. **Issue #290**: Episodic Data Abstractions (N-way K-shot task sampling) - **COMPLETED**
2. **Issue #292**: Reptile Algorithm (simple first-order meta-learning) - **COMPLETED**
3. **Issue #291**: MAML Algorithm (gradient-based meta-learning)
4. **Issue #289**: SEAL Algorithm (self-supervised + active learning meta-learning)
5. **Issue #288**: Documentation and Examples for all meta-learning features

---

## Understanding Meta-Learning

### What is Meta-Learning?

**Meta-learning** (or "learning to learn") trains models to quickly adapt to new tasks with very few examples. Instead of training once on a large dataset, you train across many small tasks to learn how to learn efficiently.

**Traditional Learning vs Meta-Learning:**

```
Traditional Learning:
- Train on 10,000 cat/dog images
- Result: Model that classifies cats vs dogs well
- Problem: Can't classify birds without retraining on 10,000 bird images

Meta-Learning:
- Train on 100 different classification tasks (cats vs dogs, birds vs planes, etc.)
- Each task has only 5 examples per class (5-shot learning)
- Result: Model learns HOW to quickly adapt to new tasks
- Benefit: Can classify new categories from just 5 examples!
```

### Real-World Applications:

- **Few-shot image classification**: Recognize new objects from 1-5 images
- **Rapid robot adaptation**: Adapt to new environments with minimal data
- **Personalized recommendations**: Adapt to new users quickly
- **Drug discovery**: Predict properties of new molecules from limited experiments
- **Medical diagnosis**: Learn new rare diseases from few patient records

### The Two-Loop Architecture:

Meta-learning uses a **two-level optimization** structure:

```
OUTER LOOP (Meta-Optimization):
├─ Goal: Learn parameters θ that enable fast adaptation
├─ Process: Sample batch of tasks from task distribution
└─ Update: Modify θ based on how well the model adapted to tasks

    INNER LOOP (Task Adaptation):
    ├─ Goal: Quickly adapt θ to solve a specific task
    ├─ Process: Few gradient steps (1-10) on support set
    └─ Update: Create task-specific parameters θ_i
```

**Example with 5-way 5-shot classification:**

```
Outer Loop Iteration 1:
  Task 1: Classify birds (sparrow, robin, eagle, hawk, owl) - 5 examples each
    Inner Loop: 5 gradient steps on 25 examples → adapted model
    Evaluation: Test on query set (new bird images) → accuracy 85%

  Task 2: Classify vehicles (car, truck, bike, bus, train) - 5 examples each
    Inner Loop: 5 gradient steps on 25 examples → adapted model
    Evaluation: Test on query set → accuracy 78%

  Task 3: Classify fruits (apple, orange, banana, grape, pear) - 5 examples each
    Inner Loop: 5 gradient steps on 25 examples → adapted model
    Evaluation: Test on query set → accuracy 82%

  Meta-Update: Adjust θ to improve average adaptation performance

After 1000 outer loop iterations:
  Model learns an initialization that can quickly adapt to ANY new 5-way classification task!
```

---

## The Meta-Learning Algorithms

### 1. Reptile (Nichol et al., 2018) - **ALREADY IMPLEMENTED**

**Intuition**: Simply move meta-parameters toward task-adapted parameters.

**Algorithm:**
```
Initialize: θ (meta-parameters)

for each meta-iteration:
    Sample task T
    Clone θ → θ_task

    # Inner loop: Adapt to task
    for k steps:
        θ_task = θ_task - α ∇L(θ_task, support_set)

    # Outer loop: Move toward adapted parameters
    θ = θ + ε(θ_task - θ)
```

**Why it works**: By repeatedly moving toward task-adapted parameters, θ converges to a region where:
- Many tasks can be solved with few gradient steps
- The loss surface is smooth and easy to navigate
- Small parameter changes lead to effective solutions

**Advantages**:
- Simple to implement (no second-order derivatives)
- Memory-efficient
- Works well in practice
- Fast training

**Location**: `src/MetaLearning/Trainers/ReptileTrainer.cs`

### 2. MAML (Finn et al., 2017) - **TO BE IMPLEMENTED**

**Model-Agnostic Meta-Learning** - The most influential meta-learning algorithm.

**Intuition**: Find parameters θ where a few gradient steps lead to good performance across many tasks.

**Algorithm:**
```
Initialize: θ (meta-parameters)

for each meta-iteration:
    Sample batch of tasks: T1, T2, ..., Tn

    for each task Ti:
        # Inner loop: Adapt to task
        θ_i = θ
        for k steps:
            θ_i = θ_i - α ∇L(θ_i, support_set_i)

        # Evaluate adapted model
        loss_i = L(θ_i, query_set_i)

    # Outer loop: Meta-gradient through inner loop
    meta_loss = mean(loss_i for all tasks)
    θ = θ - β ∇_θ meta_loss  # Gradient through inner loop!
```

**Key Difference from Reptile**: MAML computes gradients **through** the inner loop optimization, requiring second-order derivatives.

**First-Order MAML (FOMAML)**: Ignore second-order derivatives for efficiency (similar to Reptile but evaluates on query set).

**Advantages**:
- Theoretically principled
- Explicit optimization for fast adaptation
- Often achieves better performance than Reptile

**Challenges**:
- Requires second-order derivatives (computationally expensive)
- Higher memory usage
- More complex implementation

**Location**: `src/MetaLearning/Trainers/MAMLTrainer.cs` (TO BE CREATED)

### 3. SEAL (Hao et al., 2019) - **TO BE IMPLEMENTED**

**Self-supervised and Episodic Active Learning** - Combines three powerful ideas.

**Intuition**: Use unlabeled data in the query set to improve few-shot learning through:
1. **Self-supervised pre-training** (learn representations without labels)
2. **Active learning** (select most informative examples to pseudo-label)
3. **Supervised fine-tuning** (combine real labels + pseudo-labels)

**Algorithm:**
```
Initialize: θ (meta-parameters)

for each meta-iteration:
    Sample task T with support set S (labeled) and query set Q (treat as unlabeled)
    Clone θ → θ_task

    # Phase 1: Self-supervised pre-training on Q
    for k1 steps:
        Create self-supervised task (e.g., rotation prediction)
        θ_task = θ_task - α ∇L_self_supervised(θ_task, Q)

    # Phase 2: Active learning - select most confident predictions
    predictions = θ_task(Q)
    confident_examples = top_k(predictions, by=confidence)
    pseudo_labeled_set = (confident_examples, predicted_labels)

    # Phase 3: Supervised fine-tuning on S + pseudo-labeled set
    combined_set = S ∪ pseudo_labeled_set
    for k2 steps:
        θ_task = θ_task - α ∇L_supervised(θ_task, combined_set)

    # Outer loop: Reptile-style update
    θ = θ + ε(θ_task - θ)
```

**Why it works**:
- Self-supervised learning helps model understand the data structure
- Active learning identifies examples model is confident about
- Pseudo-labels increase effective training data size
- Combining real + pseudo labels improves adaptation

**Advantages**:
- Leverages unlabeled query set data
- Often outperforms MAML and Reptile
- Particularly effective when query set is large

**Challenges**:
- More complex than MAML/Reptile
- Requires self-supervised loss function (e.g., rotation prediction)
- Need to tune multiple hyperparameters

**Location**: `src/MetaLearning/Trainers/SEALTrainer.cs` (TO BE CREATED)

---

## Understanding N-way K-shot Tasks

### What is N-way K-shot?

**N-way K-shot** defines the structure of a meta-learning task:
- **N-way**: Number of classes in the task
- **K-shot**: Number of labeled examples per class in the support set

**Example: 5-way 3-shot image classification**

```
Support Set (training data for this task):
  Class 1 (cat):      [image1, image2, image3]         (3 shots)
  Class 2 (dog):      [image4, image5, image6]         (3 shots)
  Class 3 (bird):     [image7, image8, image9]         (3 shots)
  Class 4 (car):      [image10, image11, image12]      (3 shots)
  Class 5 (tree):     [image13, image14, image15]      (3 shots)
  Total: 5 classes × 3 shots = 15 examples

Query Set (test data for this task):
  Class 1 (cat):      [image16, image17, image18, ..., image25]  (10 queries)
  Class 2 (dog):      [image26, image27, image28, ..., image35]  (10 queries)
  Class 3 (bird):     [image36, image37, image38, ..., image45]  (10 queries)
  Class 4 (car):      [image46, image47, image48, ..., image55]  (10 queries)
  Class 5 (tree):     [image56, image57, image58, ..., image65]  (10 queries)
  Total: 5 classes × 10 queries = 50 examples
```

**Task Goal**: Train on 15 support examples, then correctly classify the 50 query examples.

### Episodic Data Loader

The `EpisodicDataLoader` (Issue #290) samples these N-way K-shot tasks from a larger dataset.

**Key Features**:
- Randomly samples N classes from dataset
- Randomly samples K examples per class for support set
- Randomly samples Q examples per class for query set
- Ensures no overlap between support and query sets
- Each call to `GetNextTask()` produces a new random task

**Example Usage**:
```csharp
// Configure 5-way 5-shot with 15 queries per class
var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
    datasetX: allFeatures,        // Full dataset
    datasetY: allLabels,
    nWay: 5,                       // 5 classes per task
    kShot: 5,                      // 5 support examples per class
    queryShots: 15                 // 15 query examples per class
);

// Sample a task
var task = dataLoader.GetNextTask();

// task.SupportSetX: (25 × 784) - 25 images, 784 pixels each
// task.SupportSetY: (25 × 5) - 25 one-hot labels, 5 classes
// task.QuerySetX: (75 × 784) - 75 query images
// task.QuerySetY: (75 × 5) - 75 query labels
```

**Location**: `src/Data/Loaders/UniformEpisodicDataLoader.cs` (ALREADY IMPLEMENTED)

---

## Implementation Status

### ✅ Completed (Issues #290, #292):

1. **Episodic Data Abstractions**:
   - `IEpisodicDataLoader<T, TInput, TOutput>` interface
   - `MetaLearningTask<T, TInput, TOutput>` data structure
   - `UniformEpisodicDataLoader` (random sampling)
   - `BalancedEpisodicDataLoader` (balanced class sampling)
   - `StratifiedEpisodicDataLoader` (stratified sampling)
   - `CurriculumEpisodicDataLoader` (curriculum learning)

2. **Reptile Algorithm**:
   - `IMetaLearner<T, TInput, TOutput>` interface
   - `ReptileTrainerBase<T, TInput, TOutput>` base class
   - `ReptileTrainer<T, TInput, TOutput>` implementation
   - `ReptileTrainerConfig<T>` configuration
   - Result types: `MetaTrainingResult`, `MetaAdaptationResult`, etc.
   - Full test coverage

### 🚧 To Be Implemented:

1. **Issue #291: MAML Algorithm**:
   - `MAMLTrainer<T, TInput, TOutput>` implementation
   - First-order and second-order variants
   - Inner loop optimization with gradient tracking
   - Meta-gradient computation
   - Unit and integration tests

2. **Issue #289: SEAL Algorithm**:
   - `SEALTrainer<T, TInput, TOutput>` implementation
   - Self-supervised loss function (rotation prediction)
   - Active learning selection strategy
   - Pseudo-labeling logic
   - Three-phase training (self-supervised → active → supervised)
   - Unit and integration tests

3. **Issue #288: Documentation and Examples**:
   - `docs/MetaLearning.md` comprehensive guide
   - `testconsole/Examples/MetaLearning/ReptileExample.cs`
   - `testconsole/Examples/MetaLearning/MAMLExample.cs`
   - `testconsole/Examples/MetaLearning/SEALExample.cs`
   - Runnable examples with synthetic datasets

---

## Architectural Patterns (CRITICAL)

### 1. Generic Type Parameters

All meta-learning components use **three type parameters**:

```csharp
public interface IMetaLearner<T, TInput, TOutput>
{
    // T: Numeric type (double, float)
    // TInput: Input data type (Matrix<T>, Tensor<T>, double[])
    // TOutput: Output data type (Vector<T>, Tensor<T>, double[])
}
```

**Why three parameters?**
- `T`: Enables numeric generics (use `NumOps` for all arithmetic)
- `TInput`: Flexible input format (images, sequences, tabular data)
- `TOutput`: Flexible output format (class labels, regression values)

### 2. Configuration Objects

All trainers accept a configuration object:

```csharp
public class ReptileTrainer<T, TInput, TOutput> : ReptileTrainerBase<T, TInput, TOutput>
{
    public ReptileTrainer(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        IMetaLearnerConfig<T>? config = null)  // Optional, uses defaults if null
    {
        // ...
    }
}
```

**Benefits**:
- Beginner-friendly: Can use defaults
- Advanced: Can customize all hyperparameters
- Maintainable: Adding new parameters doesn't break existing code

### 3. Builder Pattern Integration

Meta-learners should integrate with `PredictionModelBuilder`:

```csharp
// Future integration (not yet implemented):
var model = new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
    .WithNeuralNetwork(layers)
    .WithMetaLearning(new ReptileTrainerConfig<double>(...))
    .Build();
```

### 4. Result Types

All operations return rich result objects:

```csharp
// Training step result
public class MetaTrainingStepResult<T>
{
    public T MetaLoss { get; }
    public T TaskLoss { get; }
    public T Accuracy { get; }
    public int NumTasks { get; }
    public int Iteration { get; }
    public double TimeMs { get; }
}

// Full training result
public class MetaTrainingResult<T>
{
    public T FinalLoss { get; }
    public T BestLoss { get; }
    public TimeSpan TrainingTime { get; }
    public List<T> LossHistory { get; }
    public List<T> AccuracyHistory { get; }
}

// Adaptation result
public class MetaAdaptationResult<T>
{
    public T QueryAccuracy { get; }
    public T QueryLoss { get; }
    public T SupportAccuracy { get; }
    public T SupportLoss { get; }
    public int AdaptationSteps { get; }
    public double AdaptationTimeMs { get; }
    public List<T> PerStepLosses { get; }
}
```

---

## Testing Strategy

### 1. Synthetic Datasets

**Sine Wave Regression** (for MAML/Reptile):
```csharp
// Each task: Fit a sine wave with different amplitude/phase
// Support set: 10 (x, y) points
// Query set: 50 (x, y) points
// Goal: After meta-training, adapt to new sine waves quickly
```

**Rotated MNIST** (for SEAL):
```csharp
// Each task: Classify digits at different rotations
// Support set: 5 images per digit (1-shot or 5-shot)
// Query set: 50 images per digit (unlabeled for SEAL)
// Self-supervised task: Predict rotation angle (0°, 90°, 180°, 270°)
```

### 2. Test Metrics

**Before meta-training** (baseline):
- Average query loss on unseen tasks: ~2.5
- Average query accuracy: ~20% (random guess for 5-way)

**After meta-training** (expected):
- Average query loss: <0.5
- Average query accuracy: >70%
- Adaptation should be visible after just 5 inner loop steps

**Success Criteria**:
```csharp
[Fact]
public void MetaLearning_ImprovesFewShotPerformance()
{
    // Pre-training evaluation
    double baselineAccuracy = EvaluateOnTasks(metaModel, testTasks);

    // Meta-training
    metaLearner.Train();

    // Post-training evaluation
    double metaAccuracy = EvaluateOnTasks(metaModel, testTasks);

    // Assert significant improvement
    Assert.True(metaAccuracy > baselineAccuracy + 0.4);  // At least 40% improvement
    Assert.True(metaAccuracy > 0.70);  // At least 70% absolute accuracy
}
```

---

## Common Pitfalls to Avoid

### 1. Confusing Support vs Query Sets

❌ **WRONG**:
```csharp
// Training on query set
for (int step = 0; step < innerSteps; step++)
{
    model.Train(task.QuerySetX, task.QuerySetY);  // WRONG!
}
```

✅ **CORRECT**:
```csharp
// Inner loop: Train on SUPPORT set
for (int step = 0; step < innerSteps; step++)
{
    model.Train(task.SupportSetX, task.SupportSetY);
}

// Evaluation: Test on QUERY set
double loss = ComputeLoss(model, task.QuerySetX, task.QuerySetY);
```

### 2. Forgetting to Clone Parameters

❌ **WRONG**:
```csharp
// All tasks modify the same model!
for (int i = 0; i < batchSize; i++)
{
    var task = dataLoader.GetNextTask();
    model.Train(task.SupportSetX, task.SupportSetY);  // Modifies shared model
}
```

✅ **CORRECT**:
```csharp
// Each task gets its own clone
Vector<T> originalParams = metaModel.GetParameters();

for (int i = 0; i < batchSize; i++)
{
    var task = dataLoader.GetNextTask();

    // Reset to original parameters for each task
    metaModel.SetParameters(originalParams.Clone());

    // Now adapt this clone
    metaModel.Train(task.SupportSetX, task.SupportSetY);
}
```

### 3. Using Wrong Learning Rates

Meta-learning has **two learning rates**:

```csharp
// Inner loop (task adaptation) - typically LARGER
double innerLearningRate = 0.01;  // Fast adaptation

// Outer loop (meta-update) - typically SMALLER
double metaLearningRate = 0.001;  // Slow meta-learning
```

**Why different rates?**
- Inner loop: Quick adaptation to specific task (few steps, large steps)
- Outer loop: Gradual improvement of meta-parameters (many iterations, small steps)

### 4. Not Tracking Gradients in MAML

MAML requires gradients **through** the inner loop:

❌ **WRONG** (First-order approximation only):
```csharp
// Inner loop
for (int step = 0; step < innerSteps; step++)
{
    model.Train(task.SupportSetX, task.SupportSetY);
    // Gradients are discarded after each step
}

// Meta-gradient only from query set evaluation
double metaLoss = ComputeLoss(model, task.QuerySetX, task.QuerySetY);
```

✅ **CORRECT** (Full MAML):
```csharp
// Inner loop with gradient tracking
var computationGraph = new List<GradientNode>();

for (int step = 0; step < innerSteps; step++)
{
    // Track gradients through inner loop
    var gradNode = model.TrainWithGradientTracking(task.SupportSetX, task.SupportSetY);
    computationGraph.Add(gradNode);
}

// Meta-gradient backpropagates through entire computation graph
double metaLoss = ComputeLoss(model, task.QuerySetX, task.QuerySetY);
Backpropagate(metaLoss, throughGraph: computationGraph);
```

### 5. Incorrect SEAL Phase Ordering

SEAL must follow the correct sequence:

❌ **WRONG**:
```csharp
// Supervised first (defeats the purpose of self-supervision)
model.Train(task.SupportSetX, task.SupportSetY);
SelfSupervisedPretraining(model, task.QuerySetX);
```

✅ **CORRECT**:
```csharp
// Phase 1: Self-supervised pre-training (learn representations)
SelfSupervisedPretraining(model, task.QuerySetX);

// Phase 2: Active learning (select confident examples)
var pseudoLabeled = ActiveLearningSelection(model, task.QuerySetX);

// Phase 3: Supervised fine-tuning (combine real + pseudo labels)
var combinedSet = Combine(task.SupportSetX, task.SupportSetY, pseudoLabeled);
model.Train(combinedSet.X, combinedSet.Y);
```

---

## Next Steps

### For Implementers:

1. **Read Issue #291 (MAML)**: Understand gradient-through-gradient computation
2. **Read Issue #289 (SEAL)**: Understand self-supervised learning and active learning
3. **Read Issue #288 (Documentation)**: See what examples are needed
4. **Review existing code**:
   - `src/MetaLearning/Trainers/ReptileTrainer.cs` (reference implementation)
   - `src/Interfaces/IMetaLearner.cs` (interface to implement)
   - `src/Data/Loaders/UniformEpisodicDataLoader.cs` (how tasks are sampled)

### For This Epic (Issue #286):

**This issue is NOT directly implementable** - it's a tracking epic. To make progress:

1. ✅ **Completed**: Issue #290 (Episodic Data Abstractions)
2. ✅ **Completed**: Issue #292 (Reptile Baseline)
3. ⏳ **In Progress**: Issue #291 (MAML Baseline)
4. ⏳ **In Progress**: Issue #289 (SEAL Algorithm)
5. ⏳ **Pending**: Issue #288 (Documentation and Examples)

Once all 5 sub-issues are complete, this epic can be closed.

---

## References

### Papers:
- **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
- **Reptile**: Nichol et al., "On First-Order Meta-Learning Algorithms", arXiv 2018
- **SEAL**: Hao et al., "Self-Ensembling for Visual Domain Adaptation and Semi-Supervised Meta-Learning", ICLR 2019

### Existing Implementation Files:
- `src/Interfaces/IMetaLearner.cs`
- `src/Interfaces/IMetaLearnerConfig.cs`
- `src/Interfaces/IEpisodicDataLoader.cs`
- `src/Data/Abstractions/MetaLearningTask.cs`
- `src/Data/Loaders/UniformEpisodicDataLoader.cs`
- `src/MetaLearning/Trainers/ReptileTrainerBase.cs`
- `src/MetaLearning/Trainers/ReptileTrainer.cs`
- `src/MetaLearning/Config/ReptileTrainerConfig.cs`
- `src/Models/Results/MetaTrainingResult.cs`
- `src/Models/Results/MetaAdaptationResult.cs`
- `src/Models/Results/MetaEvaluationResult.cs`
- `src/Models/Results/MetaTrainingStepResult.cs`

### Test Files:
- `tests/UnitTests/MetaLearning/ReptileTrainerTests.cs`
- `tests/UnitTests/MetaLearning/ReptileTrainerIntegrationTests.cs`
- `tests/UnitTests/Data/UniformEpisodicDataLoaderTests.cs`
