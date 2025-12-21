# Implementation Plan: PR#430 Production Readiness

## Overview

This document outlines the comprehensive implementation plan to address all findings from the PR#430 audit. Every item must be implemented with production-ready code before the PR can be considered complete.

**Total Items to Address: 13**
- Missing Implementations: 3
- Simplified Code Requiring Production-Ready Fixes: 10

---

## Part 1: Missing Implementations

### 1.1 Progressive Neural Networks (PNN) Strategy

**Priority: HIGH**
**Estimated Complexity: High**

#### Current State
- Configuration parameters exist in `ContinualLearnerConfig.cs` (lines 156-162):
  - `PnnUseLateralConnections`
  - `PnnLateralScaling`
- **No actual strategy implementation exists**

#### Required Implementation

**File to Create:** `src/ContinualLearning/Strategies/ProgressiveNeuralNetworksStrategy.cs`

**Algorithm Description:**
Progressive Neural Networks freeze previous task columns and add new columns with lateral connections for each new task. This prevents catastrophic forgetting by preserving old knowledge while allowing new learning.

**Key Components to Implement:**

1. **Column Management**
   - Maintain list of frozen neural network columns (one per task)
   - Add new trainable column for each new task
   - Freeze previous columns when task completes

2. **Lateral Connections**
   - Implement adapter layers connecting previous columns to new column
   - Apply lateral scaling factor from config
   - Forward pass must aggregate activations from all columns

3. **Training Logic**
   - Only train parameters in the newest column
   - Lateral connection weights are trainable
   - Previous column weights remain frozen

**Reference Paper:** Rusu et al., "Progressive Neural Networks" (2016)

**Interface to Implement:**
```csharp
public class ProgressiveNeuralNetworksStrategy<T, TInput, TOutput>
    : ContinualStrategyBase<T, TInput, TOutput>, IProgressiveStrategy<T, TInput, TOutput>
{
    // Column storage
    private readonly List<ILayer<T>[]> _frozenColumns;
    private ILayer<T>[]? _activeColumn;
    private readonly List<Matrix<T>[]> _lateralConnections;

    // Core methods
    public override void BeforeTaskTraining(int taskId, IDataset<T, TInput, TOutput> taskData);
    public override void AfterTaskTraining(int taskId, IFullModel<T, TInput, TOutput> model);
    public override Tensor<T> Forward(Tensor<T> input, int taskId);
    public void FreezeCurrentColumn();
    public void AddNewColumn(int[] layerSizes);
    public Matrix<T> ComputeLateralActivations(int columnIndex, int layerIndex, Tensor<T> input);
}
```

---

### 1.2 PackNet Strategy

**Priority: HIGH**
**Estimated Complexity: High**

#### Current State
- Configuration parameters exist in `ContinualLearnerConfig.cs` (lines 146-152):
  - `PackNetPruneRatio`
  - `PackNetRetrainEpochs`
- **No actual strategy implementation exists**

#### Required Implementation

**File to Create:** `src/ContinualLearning/Strategies/PackNetStrategy.cs`

**Algorithm Description:**
PackNet iteratively prunes and freezes network weights after each task, freeing capacity for new tasks while preserving performance on old tasks.

**Key Components to Implement:**

1. **Weight Masking System**
   - Binary masks for each layer indicating which weights are "owned" by which task
   - Cumulative mask tracking all frozen weights
   - Available capacity mask for new task training

2. **Pruning Algorithm**
   - Magnitude-based pruning (prune smallest weights)
   - Configurable prune ratio per task
   - Preserve minimum weights needed for task performance

3. **Retraining Phase**
   - After pruning, retrain remaining weights
   - Only train weights not frozen by previous tasks
   - Validate performance doesn't degrade

4. **Inference Logic**
   - Apply appropriate mask for each task during inference
   - Support multi-task inference with task ID

**Reference Paper:** Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning" (CVPR 2018)

**Interface to Implement:**
```csharp
public class PackNetStrategy<T, TInput, TOutput>
    : ContinualStrategyBase<T, TInput, TOutput>, IPackNetStrategy<T, TInput, TOutput>
{
    // Mask storage
    private readonly Dictionary<int, List<Tensor<T>>> _taskMasks; // Task ID -> layer masks
    private readonly List<Tensor<T>> _frozenMasks; // Cumulative frozen weights

    // Core methods
    public override void AfterTaskTraining(int taskId, IFullModel<T, TInput, TOutput> model);
    public void PruneNetwork(IFullModel<T, TInput, TOutput> model, T pruneRatio);
    public void FreezeTaskWeights(int taskId);
    public Tensor<T> GetAvailableCapacityMask(int layerIndex);
    public void ApplyMaskForTask(IFullModel<T, TInput, TOutput> model, int taskId);
    public void RetrainAfterPruning(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> data, int epochs);
}
```

---

### 1.3 Expected Gradient Length (EGL) Strategy

**Priority: HIGH**
**Estimated Complexity: Medium**

#### Current State
- **Completely missing from Active Learning**
- No configuration, no implementation

#### Required Implementation

**File to Create:** `src/ActiveLearning/Strategies/ExpectedGradientLengthStrategy.cs`

**Algorithm Description:**
EGL selects samples that would cause the largest gradient update if labeled. It estimates the expected gradient length across possible labels, selecting samples that would most change the model.

**Key Components to Implement:**

1. **Gradient Computation**
   - Compute gradients for each possible label
   - Use model's actual loss function (not simplified MSE)
   - Handle multi-class and regression cases

2. **Expected Length Calculation**
   - Weight gradients by predicted probability of each label
   - Compute L2 norm of expected gradient
   - Normalize across samples for fair comparison

3. **Efficient Implementation**
   - Batch gradient computation where possible
   - Cache intermediate computations
   - Support for models with `IGradientComputable` interface

**Reference Paper:** Settles & Craven, "An Analysis of Active Learning Strategies for Sequence Labeling Tasks" (EMNLP 2008)

**Interface to Implement:**
```csharp
public class ExpectedGradientLengthStrategy<T, TInput, TOutput>
    : IActiveLearningStrategy<T, TInput, TOutput>
{
    public string Name => "Expected Gradient Length";

    // Core methods
    public Vector<T> ComputeScores(
        IDataset<T, TInput, TOutput> unlabeledPool,
        IFullModel<T, TInput, TOutput> model);

    public int[] SelectSamples(
        IDataset<T, TInput, TOutput> unlabeledPool,
        IFullModel<T, TInput, TOutput> model,
        int count);

    // Internal methods
    private T ComputeExpectedGradientLength(
        TInput input,
        IFullModel<T, TInput, TOutput> model);

    private Vector<T> ComputeGradientForLabel(
        TInput input,
        TOutput hypotheticalLabel,
        IFullModel<T, TInput, TOutput> model);
}
```

---

## Part 2: Simplified Code Requiring Production-Ready Fixes

### 2.1 ExperienceReplayBuffer - HerdingSample

**File:** `src/ContinualLearning/Memory/ExperienceReplayBuffer.cs`
**Line:** 420
**Current Issue:** "Simplified herding: select diverse examples using hash-based diversity"

#### Current Code Problem
Uses `GetHashCode()` for diversity measurement instead of proper feature extraction and mean matching.

#### Required Fix

**Proper Herding Algorithm:**
1. Extract feature representations from model's penultimate layer
2. Compute running mean of selected samples
3. Iteratively select sample that moves mean closest to population mean
4. Use actual feature distances, not hash codes

```csharp
private List<ReplayExperience<T, TInput, TOutput>> HerdingSample(int count)
{
    var selected = new List<ReplayExperience<T, TInput, TOutput>>();
    var remaining = new List<int>(Enumerable.Range(0, _buffer.Count));

    // Extract features for all samples (requires model access)
    var features = ExtractFeatures(_buffer);

    // Compute population mean
    var populationMean = ComputeMean(features);

    // Running sum of selected features
    var selectedSum = new Vector<T>(features[0].Length);

    for (int i = 0; i < count && remaining.Count > 0; i++)
    {
        int bestIdx = -1;
        T bestDistance = NumOps.MaxValue;

        foreach (var idx in remaining)
        {
            // Compute mean if we add this sample
            var newSum = VectorAdd(selectedSum, features[idx]);
            var newMean = VectorDivide(newSum, NumOps.FromDouble(i + 1));

            // Distance to population mean
            var distance = ComputeL2Distance(newMean, populationMean);

            if (NumOps.Compare(distance, bestDistance) < 0)
            {
                bestDistance = distance;
                bestIdx = idx;
            }
        }

        if (bestIdx >= 0)
        {
            selected.Add(_buffer[bestIdx]);
            selectedSum = VectorAdd(selectedSum, features[bestIdx]);
            remaining.Remove(bestIdx);
        }
    }

    return selected;
}
```

---

### 2.2 ExperienceReplayBuffer - KCenterSample

**File:** `src/ContinualLearning/Memory/ExperienceReplayBuffer.cs`
**Line:** 449
**Current Issue:** "Simplified K-Center greedy: use hash-based distance approximation"

#### Current Code Problem
Uses hash codes as distance proxy instead of actual feature-space distances.

#### Required Fix

**Proper K-Center Greedy Algorithm:**
1. Extract feature representations for all samples
2. Initialize with random or farthest-first sample
3. Iteratively select sample farthest from all selected samples
4. Use actual L2 or cosine distance in feature space

```csharp
private List<ReplayExperience<T, TInput, TOutput>> KCenterSample(int count)
{
    if (_buffer.Count <= count)
        return new List<ReplayExperience<T, TInput, TOutput>>(_buffer);

    var selected = new List<int>();
    var features = ExtractFeatures(_buffer);

    // Start with random sample
    selected.Add(RandomHelper.Shared.Next(_buffer.Count));

    // Track minimum distance to any selected sample for each point
    var minDistances = new T[_buffer.Count];
    for (int i = 0; i < _buffer.Count; i++)
    {
        minDistances[i] = NumOps.MaxValue;
    }

    while (selected.Count < count)
    {
        int lastSelected = selected[^1];

        // Update minimum distances
        for (int i = 0; i < _buffer.Count; i++)
        {
            if (selected.Contains(i)) continue;

            var dist = ComputeL2Distance(features[i], features[lastSelected]);
            if (NumOps.Compare(dist, minDistances[i]) < 0)
            {
                minDistances[i] = dist;
            }
        }

        // Select point with maximum minimum distance (farthest from all selected)
        int bestIdx = -1;
        T maxMinDist = NumOps.MinValue;

        for (int i = 0; i < _buffer.Count; i++)
        {
            if (selected.Contains(i)) continue;

            if (NumOps.Compare(minDistances[i], maxMinDist) > 0)
            {
                maxMinDist = minDistances[i];
                bestIdx = i;
            }
        }

        if (bestIdx >= 0)
        {
            selected.Add(bestIdx);
        }
    }

    return selected.Select(i => _buffer[i]).ToList();
}
```

---

### 2.3 MemoryAwareSynapses - ComputeRandomProjectionImportance

**File:** `src/ContinualLearning/Strategies/MemoryAwareSynapses.cs`
**Line:** 527-529
**Current Issue:** Just falls back to `ComputeOutputSensitivity`

#### Required Fix

**Proper Random Projection Implementation:**
1. Generate stable random projection matrix
2. Project parameter gradients onto random directions
3. Compute importance as gradient magnitude in projected space

```csharp
private Vector<T> ComputeRandomProjectionImportance(
    IFullModel<T, TInput, TOutput> model,
    IDataset<T, TInput, TOutput> dataset)
{
    var parameters = model.GetParameters();
    int paramCount = parameters.Length;
    int projectionDim = Math.Min(paramCount, 100); // Reduced dimensionality

    // Generate stable random projection matrix (seeded for reproducibility)
    var projectionMatrix = GenerateRandomProjectionMatrix(paramCount, projectionDim, seed: 42);

    var importanceAccumulator = new Vector<T>(paramCount);

    foreach (var (input, output) in dataset.GetBatches(1))
    {
        // Compute gradients
        var gradients = ComputeParameterGradients(model, input, output);

        // Project gradients
        var projectedGrad = MatrixVectorMultiply(projectionMatrix, gradients);

        // Backproject to get importance estimate
        var backprojected = MatrixTransposeVectorMultiply(projectionMatrix, projectedGrad);

        // Accumulate squared importance
        for (int i = 0; i < paramCount; i++)
        {
            var squared = NumOps.Multiply(backprojected[i], backprojected[i]);
            importanceAccumulator[i] = NumOps.Add(importanceAccumulator[i], squared);
        }
    }

    // Normalize by dataset size
    var scale = NumOps.FromDouble(1.0 / dataset.Count);
    return VectorScale(importanceAccumulator, scale);
}
```

---

### 2.4 MemoryAwareSynapses - ComputeFisherDiagonalImportance

**File:** `src/ContinualLearning/Strategies/MemoryAwareSynapses.cs`
**Line:** 535-540
**Current Issue:** "Fall back for now"

#### Required Fix

**Proper Fisher Information Diagonal:**
1. Compute log-likelihood gradients for each sample
2. Square the gradients (Fisher = E[grad log p * grad log p^T])
3. Average across dataset for diagonal Fisher approximation

```csharp
private Vector<T> ComputeFisherDiagonalImportance(
    IFullModel<T, TInput, TOutput> model,
    IDataset<T, TInput, TOutput> dataset)
{
    var parameters = model.GetParameters();
    int paramCount = parameters.Length;
    var fisherDiagonal = new Vector<T>(paramCount);

    foreach (var (input, output) in dataset.GetBatches(1))
    {
        // Get model prediction (for log-likelihood gradient)
        var prediction = model.Predict(input);

        // Compute gradient of log-likelihood w.r.t. parameters
        // For classification: grad log p(y|x,θ)
        // For regression: grad log p(y|x,θ) under Gaussian assumption
        var logLikelihoodGradients = ComputeLogLikelihoodGradients(model, input, output, prediction);

        // Fisher diagonal is E[g * g^T] diagonal = E[g^2]
        for (int i = 0; i < paramCount; i++)
        {
            var gradSquared = NumOps.Multiply(logLikelihoodGradients[i], logLikelihoodGradients[i]);
            fisherDiagonal[i] = NumOps.Add(fisherDiagonal[i], gradSquared);
        }
    }

    // Average over dataset
    var scale = NumOps.FromDouble(1.0 / dataset.Count);
    return VectorScale(fisherDiagonal, scale);
}

private Vector<T> ComputeLogLikelihoodGradients(
    IFullModel<T, TInput, TOutput> model,
    TInput input,
    TOutput target,
    TOutput prediction)
{
    if (model is IGradientComputable<T, Tensor<T>, Tensor<T>> gradModel)
    {
        // Use cross-entropy loss for classification (gives log-likelihood gradient)
        var inputTensor = ConvertToTensor(input);
        var targetTensor = ConvertToTensor(target);
        return gradModel.ComputeGradients(inputTensor, targetTensor);
    }

    // Numerical gradient fallback
    return ComputeNumericalLogLikelihoodGradients(model, input, target);
}
```

---

### 2.5 MemoryAwareSynapses - ComputeHebbianImportance

**File:** `src/ContinualLearning/Strategies/MemoryAwareSynapses.cs`
**Line:** 545-549
**Current Issue:** "Fall back for now"

#### Required Fix

**Proper Hebbian Importance:**
1. Track co-activation patterns between connected neurons
2. Importance = strength of learned associations (Hebb's rule)
3. Weights that fire together frequently are more important

```csharp
private Vector<T> ComputeHebbianImportance(
    IFullModel<T, TInput, TOutput> model,
    IDataset<T, TInput, TOutput> dataset)
{
    var parameters = model.GetParameters();
    int paramCount = parameters.Length;
    var hebbianImportance = new Vector<T>(paramCount);

    if (model is INeuralNetworkModel<T> nnModel)
    {
        var layers = nnModel.GetLayers();
        int paramOffset = 0;

        foreach (var layer in layers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            // Compute average activations for this layer
            var preActivations = new List<Vector<T>>();
            var postActivations = new List<Vector<T>>();

            foreach (var (input, _) in dataset.GetBatches(1))
            {
                var (pre, post) = GetLayerActivations(nnModel, layer, input);
                preActivations.Add(pre);
                postActivations.Add(post);
            }

            // Hebbian importance: correlation between pre and post activations
            // For weight w_ij: importance = E[pre_i * post_j]
            var layerImportance = ComputeHebbianForLayer(
                layerParams, preActivations, postActivations);

            for (int i = 0; i < layerParamCount; i++)
            {
                hebbianImportance[paramOffset + i] = layerImportance[i];
            }

            paramOffset += layerParamCount;
        }
    }
    else
    {
        // For non-neural network models, fall back to output sensitivity
        return ComputeOutputSensitivity(model, dataset);
    }

    return hebbianImportance;
}

private Vector<T> ComputeHebbianForLayer(
    Vector<T> weights,
    List<Vector<T>> preActivations,
    List<Vector<T>> postActivations)
{
    var importance = new Vector<T>(weights.Length);
    int n = preActivations.Count;

    // Assuming weight matrix of shape [out_features, in_features]
    // Weight at position (i,j) connects pre[j] to post[i]
    int outDim = postActivations[0].Length;
    int inDim = preActivations[0].Length;

    for (int sample = 0; sample < n; sample++)
    {
        for (int i = 0; i < outDim; i++)
        {
            for (int j = 0; j < inDim; j++)
            {
                int weightIdx = i * inDim + j;
                if (weightIdx < weights.Length)
                {
                    // Hebbian: pre * post
                    var hebbian = NumOps.Multiply(
                        preActivations[sample][j],
                        postActivations[sample][i]);
                    // Use absolute value as importance
                    var absHebbian = NumOps.Abs(hebbian);
                    importance[weightIdx] = NumOps.Add(importance[weightIdx], absHebbian);
                }
            }
        }
    }

    // Average over samples
    var scale = NumOps.FromDouble(1.0 / n);
    return VectorScale(importance, scale);
}
```

---

### 2.6 SynapticIntelligence - ComputeLayerStatistics

**File:** `src/ContinualLearning/Strategies/SynapticIntelligence.cs`
**Line:** 555
**Current Issue:** "This is a simplified version - in practice, you'd need layer boundary info"

#### Required Fix

**Proper Layer-Aware Statistics:**
1. Access actual layer structure from model
2. Compute statistics per layer, not arbitrary chunks
3. Include layer type information in statistics

```csharp
private Dictionary<string, object> ComputeLayerStatistics(Vector<T> importance)
{
    var stats = new Dictionary<string, object>();

    if (_model is INeuralNetworkModel<T> nnModel)
    {
        var layers = nnModel.GetLayers();
        int paramOffset = 0;
        var layerStats = new List<Dictionary<string, object>>();

        foreach (var layer in layers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (layerParamCount == 0)
            {
                paramOffset += layerParamCount;
                continue;
            }

            // Extract importance values for this layer
            var layerImportance = new T[layerParamCount];
            for (int i = 0; i < layerParamCount; i++)
            {
                layerImportance[i] = importance[paramOffset + i];
            }

            // Compute layer statistics
            var layerStat = new Dictionary<string, object>
            {
                ["LayerName"] = layer.Name,
                ["LayerType"] = layer.GetType().Name,
                ["ParameterCount"] = layerParamCount,
                ["MeanImportance"] = NumOps.ToDouble(ComputeMean(layerImportance)),
                ["MaxImportance"] = NumOps.ToDouble(ComputeMax(layerImportance)),
                ["MinImportance"] = NumOps.ToDouble(ComputeMin(layerImportance)),
                ["StdImportance"] = NumOps.ToDouble(ComputeStd(layerImportance)),
                ["SparsityRatio"] = ComputeSparsityRatio(layerImportance, threshold: 1e-6)
            };

            layerStats.Add(layerStat);
            paramOffset += layerParamCount;
        }

        stats["LayerStatistics"] = layerStats;
        stats["TotalLayers"] = layers.Count;
        stats["TotalParameters"] = importance.Length;
    }
    else
    {
        // For non-neural network models, provide aggregate statistics only
        stats["MeanImportance"] = NumOps.ToDouble(ComputeMean(importance.ToArray()));
        stats["MaxImportance"] = NumOps.ToDouble(ComputeMax(importance.ToArray()));
        stats["TotalParameters"] = importance.Length;
        stats["Note"] = "Model does not expose layer structure";
    }

    return stats;
}

private double ComputeSparsityRatio(T[] values, double threshold)
{
    int sparseCount = values.Count(v =>
        NumOps.Compare(NumOps.Abs(v), NumOps.FromDouble(threshold)) < 0);
    return (double)sparseCount / values.Length;
}
```

---

### 2.7 CoreSetStrategy - ComputeDensityWeights

**File:** `src/ActiveLearning/Strategies/CoreSetStrategy.cs`
**Line:** 184-187
**Current Issue:** Returns equal weights instead of actual density-based weights

#### Required Fix

**Proper Density-Based Weighting:**
1. Extract features from samples
2. Compute local density using k-nearest neighbors
3. Weight samples inversely to density (rare samples are more valuable)

```csharp
private Vector<T> ComputeDensityWeights(
    IDataset<T, TInput, TOutput> pool,
    IFullModel<T, TInput, TOutput>? model)
{
    int n = pool.Count;
    var weights = new T[n];

    // Extract features
    var features = new Vector<T>[n];
    for (int i = 0; i < n; i++)
    {
        features[i] = ExtractFeatures(pool.GetInput(i), model);
    }

    // Compute k-NN density for each sample
    int k = Math.Min(10, n - 1); // k neighbors for density estimation

    for (int i = 0; i < n; i++)
    {
        // Find k nearest neighbors
        var distances = new List<(int idx, T dist)>();
        for (int j = 0; j < n; j++)
        {
            if (i == j) continue;
            var dist = ComputeL2Distance(features[i], features[j]);
            distances.Add((j, dist));
        }

        // Sort by distance and take k smallest
        distances.Sort((a, b) => NumOps.Compare(a.dist, b.dist));
        var kNearest = distances.Take(k).ToList();

        // Density = 1 / (average distance to k neighbors)
        var avgDist = NumOps.Zero;
        foreach (var (_, dist) in kNearest)
        {
            avgDist = NumOps.Add(avgDist, dist);
        }
        avgDist = NumOps.Divide(avgDist, NumOps.FromDouble(k));

        // Add small epsilon to avoid division by zero
        var epsilon = NumOps.FromDouble(1e-10);
        avgDist = NumOps.Add(avgDist, epsilon);

        // Weight inversely proportional to density
        // Low density (isolated samples) get high weight
        weights[i] = avgDist; // avgDist is already inverse of density
    }

    // Normalize weights to sum to 1
    var weightSum = NumOps.Zero;
    foreach (var w in weights)
    {
        weightSum = NumOps.Add(weightSum, w);
    }

    for (int i = 0; i < n; i++)
    {
        weights[i] = NumOps.Divide(weights[i], weightSum);
    }

    return new Vector<T>(weights);
}

private Vector<T> ExtractFeatures(TInput input, IFullModel<T, TInput, TOutput>? model)
{
    // Try to get features from model
    if (model is IFeatureExtractor<T, TInput> featureExtractor)
    {
        return featureExtractor.ExtractFeatures(input);
    }

    // Fall back to using prediction as features
    if (model != null)
    {
        var prediction = model.Predict(input);
        return ConvertToVector(prediction);
    }

    // Last resort: convert input directly
    return ConvertToVector(input);
}
```

---

### 2.8 ActiveLearner - ComputeSampleLoss

**File:** `src/ActiveLearning/Core/ActiveLearner.cs`
**Line:** 642
**Current Issue:** "This is a simplified version - real implementations would use the model's loss function"

#### Required Fix

**Use Model's Actual Loss Function:**
1. Check if model exposes its loss function
2. Use appropriate loss for model type (cross-entropy, MSE, etc.)
3. Support custom loss functions

```csharp
private T ComputeSampleLoss(
    TInput input,
    TOutput expectedOutput,
    IFullModel<T, TInput, TOutput> model)
{
    var prediction = model.Predict(input);

    // Try to use model's native loss function
    if (model is ILossComputable<T, TOutput> lossModel)
    {
        return lossModel.ComputeLoss(prediction, expectedOutput);
    }

    // Determine appropriate loss based on output type
    if (prediction is Vector<T> predVec && expectedOutput is Vector<T> targetVec)
    {
        // Check if this looks like classification (softmax output)
        if (IsClassificationOutput(predVec))
        {
            return ComputeCrossEntropyLoss(predVec, targetVec);
        }
        else
        {
            return ComputeMSELoss(predVec, targetVec);
        }
    }

    // Scalar output - use squared error
    if (prediction is T predScalar && expectedOutput is T targetScalar)
    {
        var diff = NumOps.Subtract(predScalar, targetScalar);
        return NumOps.Multiply(diff, diff);
    }

    // Fallback to generic comparison
    return ComputeGenericLoss(prediction, expectedOutput);
}

private bool IsClassificationOutput(Vector<T> output)
{
    // Classification outputs typically sum to ~1 (softmax)
    var sum = NumOps.Zero;
    foreach (var v in output)
    {
        sum = NumOps.Add(sum, v);
    }
    var diff = NumOps.Subtract(sum, NumOps.One);
    return NumOps.Compare(NumOps.Abs(diff), NumOps.FromDouble(0.1)) < 0;
}

private T ComputeCrossEntropyLoss(Vector<T> prediction, Vector<T> target)
{
    var loss = NumOps.Zero;
    var epsilon = NumOps.FromDouble(1e-15);

    for (int i = 0; i < prediction.Length; i++)
    {
        // Clamp prediction to avoid log(0)
        var clampedPred = NumOps.Compare(prediction[i], epsilon) > 0
            ? prediction[i]
            : epsilon;

        // -target * log(prediction)
        var logPred = NumOps.Log(clampedPred);
        var term = NumOps.Multiply(target[i], logPred);
        loss = NumOps.Subtract(loss, term);
    }

    return loss;
}

private T ComputeMSELoss(Vector<T> prediction, Vector<T> target)
{
    var loss = NumOps.Zero;
    int length = Math.Min(prediction.Length, target.Length);

    for (int i = 0; i < length; i++)
    {
        var diff = NumOps.Subtract(prediction[i], target[i]);
        loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
    }

    return NumOps.Divide(loss, NumOps.FromDouble(length));
}

private T ComputeGenericLoss(TOutput prediction, TOutput target)
{
    // Try to convert to vectors and compute MSE
    var predVec = ConvertToVector(prediction);
    var targetVec = ConvertToVector(target);
    return ComputeMSELoss(predVec, targetVec);
}
```

---

### 2.9 CurriculumLearner - Evaluate Method

**File:** `src/CurriculumLearning/CurriculumLearner.cs`
**Line:** 520
**Current Issue:** "simplified - assumes comparable outputs" using direct .Equals()

#### Required Fix

**Proper Evaluation with Tolerance and Type-Awareness:**
1. Use appropriate comparison for output type
2. Support classification accuracy (argmax comparison)
3. Support regression with configurable tolerance

```csharp
private CurriculumEvaluationResult<T> Evaluate(
    IFullModel<T, TInput, TOutput> model,
    IDataset<T, TInput, TOutput> dataset)
{
    if (dataset.Count == 0)
    {
        return new CurriculumEvaluationResult<T>
        {
            Accuracy = NumOps.Zero,
            Loss = NumOps.Zero,
            SampleCount = 0
        };
    }

    int correct = 0;
    var totalLoss = NumOps.Zero;

    for (int i = 0; i < dataset.Count; i++)
    {
        var input = dataset.GetInput(i);
        var expectedOutput = dataset.GetOutput(i);
        var prediction = model.Predict(input);

        // Compute loss
        var sampleLoss = ComputeLoss(prediction, expectedOutput);
        totalLoss = NumOps.Add(totalLoss, sampleLoss);

        // Check correctness
        if (IsCorrectPrediction(prediction, expectedOutput))
        {
            correct++;
        }
    }

    var accuracy = NumOps.FromDouble((double)correct / dataset.Count);
    var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(dataset.Count));

    return new CurriculumEvaluationResult<T>
    {
        Accuracy = accuracy,
        Loss = avgLoss,
        SampleCount = dataset.Count
    };
}

private bool IsCorrectPrediction(TOutput prediction, TOutput expected)
{
    // Classification: compare argmax
    if (prediction is Vector<T> predVec && expected is Vector<T> expVec)
    {
        // If looks like one-hot or probability distribution, compare argmax
        if (predVec.Length > 1 && expVec.Length > 1)
        {
            int predClass = ArgMax(predVec);
            int expClass = ArgMax(expVec);
            return predClass == expClass;
        }

        // Otherwise, use tolerance-based comparison
        return VectorsApproximatelyEqual(predVec, expVec, tolerance: 0.01);
    }

    // Regression: use tolerance
    if (prediction is T predScalar && expected is T expScalar)
    {
        var diff = NumOps.Abs(NumOps.Subtract(predScalar, expScalar));
        var tolerance = NumOps.FromDouble(0.01);
        return NumOps.Compare(diff, tolerance) < 0;
    }

    // Last resort: direct equality
    return prediction?.Equals(expected) ?? expected == null;
}

private int ArgMax(Vector<T> vec)
{
    if (vec.Length == 0) return 0;

    int maxIdx = 0;
    T maxVal = vec[0];

    for (int i = 1; i < vec.Length; i++)
    {
        if (NumOps.Compare(vec[i], maxVal) > 0)
        {
            maxVal = vec[i];
            maxIdx = i;
        }
    }

    return maxIdx;
}

private bool VectorsApproximatelyEqual(Vector<T> a, Vector<T> b, double tolerance)
{
    if (a.Length != b.Length) return false;

    var tolT = NumOps.FromDouble(tolerance);
    for (int i = 0; i < a.Length; i++)
    {
        var diff = NumOps.Abs(NumOps.Subtract(a[i], b[i]));
        if (NumOps.Compare(diff, tolT) > 0)
        {
            return false;
        }
    }

    return true;
}
```

---

### 2.10 CurriculumLearner - Logging Placeholder

**File:** `src/CurriculumLearning/CurriculumLearner.cs`
**Line:** 780
**Current Issue:** "Logs a message (placeholder for actual logging infrastructure)"

#### Required Fix

**Proper Logging Integration:**
1. Use ILogger interface for dependency injection
2. Support multiple log levels
3. Include structured logging with context

```csharp
// Add to class fields
private readonly ILogger<CurriculumLearner<T, TInput, TOutput>>? _logger;

// Update constructor to accept logger
public CurriculumLearner(
    ICurriculumScheduler<T> scheduler,
    IDifficultyEstimator<T, TInput, TOutput> difficultyEstimator,
    CurriculumLearnerConfig<T>? config = null,
    ILogger<CurriculumLearner<T, TInput, TOutput>>? logger = null)
{
    _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
    _difficultyEstimator = difficultyEstimator ?? throw new ArgumentNullException(nameof(difficultyEstimator));
    _config = config ?? new CurriculumLearnerConfig<T>();
    _logger = logger;

    // ... rest of constructor
}

// Replace Log method
private void Log(string message, LogLevel level = LogLevel.Information)
{
    if (_logger != null)
    {
        switch (level)
        {
            case LogLevel.Debug:
                _logger.LogDebug(message);
                break;
            case LogLevel.Information:
                _logger.LogInformation(message);
                break;
            case LogLevel.Warning:
                _logger.LogWarning(message);
                break;
            case LogLevel.Error:
                _logger.LogError(message);
                break;
            default:
                _logger.LogInformation(message);
                break;
        }
    }

    // Also invoke event for backward compatibility
    OnLogMessage?.Invoke(this, new LogEventArgs(message, level));
}

// Add logging event for non-DI scenarios
public event EventHandler<LogEventArgs>? OnLogMessage;

public class LogEventArgs : EventArgs
{
    public string Message { get; }
    public LogLevel Level { get; }
    public DateTime Timestamp { get; }

    public LogEventArgs(string message, LogLevel level)
    {
        Message = message;
        Level = level;
        Timestamp = DateTime.UtcNow;
    }
}
```

---

## Part 3: Implementation Order and Dependencies

### Phase 1: Foundation Fixes (Required First)
1. **2.8 ActiveLearner - ComputeSampleLoss** - Many components depend on proper loss computation
2. **2.9 CurriculumLearner - Evaluate** - Needed for accurate training metrics
3. **2.10 CurriculumLearner - Logging** - Helps debug subsequent implementations

### Phase 2: Memory and Feature Extraction
4. **2.1 ExperienceReplayBuffer - HerdingSample** - Requires feature extraction helper
5. **2.2 ExperienceReplayBuffer - KCenterSample** - Uses same feature extraction
6. **2.7 CoreSetStrategy - ComputeDensityWeights** - Similar feature extraction needs

### Phase 3: Importance Computation
7. **2.3 MemoryAwareSynapses - RandomProjection** - Independent
8. **2.4 MemoryAwareSynapses - FisherDiagonal** - Independent
9. **2.5 MemoryAwareSynapses - Hebbian** - Requires layer access
10. **2.6 SynapticIntelligence - LayerStatistics** - Requires layer access

### Phase 4: New Strategy Implementations
11. **1.3 Expected Gradient Length (EGL)** - Medium complexity, no dependencies
12. **1.1 Progressive Neural Networks (PNN)** - High complexity, requires column management
13. **1.2 PackNet Strategy** - High complexity, requires masking system

---

## Part 4: Shared Utilities to Create

Several fixes require common functionality. Create these shared utilities first:

### 4.1 Feature Extraction Helper

**File:** `src/Common/FeatureExtractionHelper.cs`

```csharp
public static class FeatureExtractionHelper<T>
{
    public static Vector<T> ExtractFeatures<TInput, TOutput>(
        TInput input,
        IFullModel<T, TInput, TOutput>? model)
    {
        // Implementation that tries multiple approaches
    }

    public static Matrix<T> ExtractBatchFeatures<TInput, TOutput>(
        IDataset<T, TInput, TOutput> dataset,
        IFullModel<T, TInput, TOutput>? model)
    {
        // Efficient batch feature extraction
    }
}
```

### 4.2 Distance Computation Helper

**File:** `src/Common/DistanceHelper.cs`

```csharp
public static class DistanceHelper<T>
{
    public static T ComputeL2Distance(Vector<T> a, Vector<T> b);
    public static T ComputeCosineDistance(Vector<T> a, Vector<T> b);
    public static T ComputeSquaredL2Distance(Vector<T> a, Vector<T> b);
    public static Matrix<T> ComputePairwiseDistances(Vector<T>[] vectors);
}
```

### 4.3 Loss Function Helper

**File:** `src/Common/LossFunctionHelper.cs`

```csharp
public static class LossFunctionHelper<T>
{
    public static T ComputeCrossEntropy(Vector<T> prediction, Vector<T> target);
    public static T ComputeMSE(Vector<T> prediction, Vector<T> target);
    public static T ComputeMAE(Vector<T> prediction, Vector<T> target);
    public static T ComputeHuberLoss(Vector<T> prediction, Vector<T> target, T delta);
}
```

---

## Part 5: Testing Requirements

Each implementation must include:

1. **Unit Tests** - Test individual methods in isolation
2. **Integration Tests** - Test interaction with real models
3. **Performance Benchmarks** - Ensure implementations are efficient
4. **Edge Case Tests** - Empty datasets, single samples, large datasets

### Test File Naming Convention
- `tests/ContinualLearning/Strategies/ProgressiveNeuralNetworksStrategyTests.cs`
- `tests/ContinualLearning/Strategies/PackNetStrategyTests.cs`
- `tests/ActiveLearning/Strategies/ExpectedGradientLengthStrategyTests.cs`

---

## Part 6: Verification Checklist

Before marking any item complete:

- [ ] Code compiles without errors or warnings
- [ ] No `// simplified`, `// placeholder`, `// TODO` comments remain
- [ ] Unit tests pass
- [ ] Integration with existing code verified
- [ ] XML documentation complete
- [ ] No hardcoded `double` or `float` (use generic `T`)
- [ ] Proper error handling with meaningful exceptions
- [ ] Thread-safety considered for parallel scenarios

---

## Summary

| Category | Items | Priority |
|----------|-------|----------|
| Missing Implementations | 3 (PNN, PackNet, EGL) | HIGH |
| Simplified Code Fixes | 10 | HIGH |
| Shared Utilities | 3 | MEDIUM (before dependent fixes) |
| **Total Work Items** | **16** | - |

**Estimated Effort:** Significant - each item requires careful implementation with proper algorithms, not quick fixes.
