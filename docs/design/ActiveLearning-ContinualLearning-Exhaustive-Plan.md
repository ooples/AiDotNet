# Active Learning & Continual Learning Exhaustive Implementation Plan

## Overview

This document outlines the exhaustive implementation plan for Active Learning (AL) and Continual Learning (CL) strategies in AiDotNet, as well as their integration into the PredictionModelBuilder facade.

## Current State

### Active Learning (5 strategies)
1. **UncertaintySampling** - Select samples where model is most uncertain
2. **QueryByCommittee** - Use committee disagreement for selection
3. **DiversitySampling** - Select diverse samples using k-center/MMD
4. **ExpectedModelChange** - Select samples causing largest gradient changes
5. **HybridSampling** - Combine uncertainty and diversity

### Continual Learning (3 strategies)
1. **ElasticWeightConsolidation (EWC)** - Fisher Information regularization
2. **GradientEpisodicMemory (GEM)** - Gradient projection with episodic memory
3. **LearningWithoutForgetting (LwF)** - Knowledge distillation from previous tasks

## Missing Strategies for Exhaustive Coverage

### Active Learning Strategies to Implement (10 new)

#### 1. RandomSampling (Baseline)
- **Purpose**: Baseline strategy for comparison
- **Method**: Random selection from unlabeled pool
- **Complexity**: O(n) for selection
- **Reference**: Standard baseline

#### 2. MarginSampling
- **Purpose**: Select samples with smallest prediction margin
- **Method**: For each sample, compute difference between top-2 probabilities
- **Formula**: margin = P(y₁|x) - P(y₂|x), select smallest margins
- **Reference**: Settles, 2012

#### 3. EntropySampling
- **Purpose**: Select samples with highest prediction entropy
- **Method**: Compute entropy of predicted class distribution
- **Formula**: H(y|x) = -Σ P(yᵢ|x) log P(yᵢ|x)
- **Reference**: Shannon, 1948; Settles, 2012

#### 4. LeastConfidenceSampling
- **Purpose**: Select samples with lowest prediction confidence
- **Method**: 1 - max(P(y|x)) for each sample
- **Reference**: Lewis & Catlett, 1994

#### 5. BALD (Bayesian Active Learning by Disagreement)
- **Purpose**: Information-theoretic sample selection
- **Method**: Mutual information between predictions and model parameters
- **Formula**: I(y; θ|x, D) = H(y|x, D) - E_θ[H(y|x, θ)]
- **Implementation**: Use MC Dropout for approximate Bayesian inference
- **Reference**: Houlsby et al., 2011

#### 6. BatchBALD
- **Purpose**: Extension of BALD for batch selection
- **Method**: Joint mutual information for batches
- **Avoids**: Redundant samples in batch
- **Reference**: Kirsch et al., 2019

#### 7. CoreSetSelection
- **Purpose**: Select representative samples covering feature space
- **Method**: k-Center-Greedy algorithm in feature space
- **Formula**: Select samples maximizing minimum distance to selected set
- **Reference**: Sener & Savarese, 2018

#### 8. DensityWeightedSampling
- **Purpose**: Combine uncertainty with density weighting
- **Method**: Score = Uncertainty × Density
- **Avoids**: Selecting outliers with high uncertainty
- **Reference**: Settles & Craven, 2008

#### 9. InformationDensity
- **Purpose**: Balance informativeness and representativeness
- **Method**: Score = Uncertainty × Avg_Similarity_to_Pool^β
- **Reference**: McCallum & Nigam, 1998

#### 10. VariationRatios
- **Purpose**: Measure prediction uncertainty via variation ratios
- **Method**: VR = 1 - max(P(y|x))
- **Interpretation**: Fraction of predicted labels not in modal class
- **Reference**: Freeman, 1965

### Continual Learning Strategies to Implement (9 new)

#### 1. SynapticIntelligence (SI)
- **Purpose**: Online importance estimation without explicit Fisher computation
- **Method**: Track parameter importance during training via path integral
- **Formula**: Ω_k = Σ_t (∂L/∂θ_k)·Δθ_k / (Δθ_k² + ξ)
- **Advantage**: More efficient than EWC (online computation)
- **Reference**: Zenke et al., 2017

#### 2. MemoryAwareSynapses (MAS)
- **Purpose**: Unsupervised importance estimation
- **Method**: Importance based on output sensitivity to weight changes
- **Formula**: Ω_ij = |∂F(x)/∂θ_ij|
- **Advantage**: No need for labeled data
- **Reference**: Aljundi et al., 2018

#### 3. PackNet
- **Purpose**: Parameter isolation via pruning
- **Method**: Prune network after each task, freeze important weights
- **Advantage**: No forgetting (hard isolation)
- **Disadvantage**: Limited capacity for many tasks
- **Reference**: Mallya & Lazebnik, 2018

#### 4. ProgressiveNeuralNetworks
- **Purpose**: Add new capacity for each task
- **Method**: Add lateral connections from frozen previous columns
- **Advantage**: No forgetting, enables forward transfer
- **Disadvantage**: Linear growth in parameters
- **Reference**: Rusu et al., 2016

#### 5. AveragedGEM (A-GEM)
- **Purpose**: Efficient approximation of GEM
- **Method**: Project gradient using random sample from memory
- **Advantage**: O(1) memory access per update (vs O(t) for GEM)
- **Reference**: Chaudhry et al., 2019

#### 6. ExperienceReplay
- **Purpose**: Rehearsal of past experiences
- **Method**: Store and replay examples from previous tasks
- **Variants**: Reservoir sampling, ring buffer, prioritized replay
- **Reference**: Rolnick et al., 2019

#### 7. GenerativeReplay
- **Purpose**: Generate pseudo-examples from previous tasks
- **Method**: Train generative model alongside main model
- **Advantage**: Constant memory regardless of task count
- **Reference**: Shin et al., 2017

#### 8. OnlineEWC
- **Purpose**: Running approximation of EWC
- **Method**: Merge Fisher information matrices across tasks
- **Formula**: F_online = γ·F_old + F_new
- **Advantage**: Constant memory (O(params) vs O(tasks × params))
- **Reference**: Schwarz et al., 2018

#### 9. VariationalContinualLearning (VCL)
- **Purpose**: Bayesian approach to continual learning
- **Method**: Maintain posterior over parameters, use as prior for new task
- **Formula**: p(θ|D₁:t) ∝ p(D_t|θ)·p(θ|D₁:t-1)
- **Reference**: Nguyen et al., 2018

## Facade Integration Design

### PredictionModelBuilder Integration

```csharp
// New private fields
private IActiveLearningStrategy<T>? _activeLearningStrategy;
private IContinualLearningStrategy<T>? _continualLearningStrategy;
private ActiveLearningOptions? _activeLearningOptions;
private ContinualLearningOptions? _continualLearningOptions;

// New configuration methods
public IPredictionModelBuilder<T, TInput, TOutput> ConfigureActiveLearning(
    IActiveLearningStrategy<T> strategy,
    ActiveLearningOptions? options = null)
{
    _activeLearningStrategy = strategy;
    _activeLearningOptions = options ?? new ActiveLearningOptions();
    return this;
}

public IPredictionModelBuilder<T, TInput, TOutput> ConfigureContinualLearning(
    IContinualLearningStrategy<T> strategy,
    ContinualLearningOptions? options = null)
{
    _continualLearningStrategy = strategy;
    _continualLearningOptions = options ?? new ContinualLearningOptions();
    return this;
}
```

### Options Classes

```csharp
/// <summary>
/// Configuration options for Active Learning.
/// </summary>
public class ActiveLearningOptions
{
    /// <summary>
    /// Number of samples to query in each active learning iteration.
    /// </summary>
    public int BatchSize { get; set; } = 10;

    /// <summary>
    /// Maximum number of active learning iterations.
    /// </summary>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Initial number of labeled samples to start with.
    /// </summary>
    public int InitialLabeledSize { get; set; } = 10;

    /// <summary>
    /// Whether to use batch diversity when selecting samples.
    /// </summary>
    public bool UseBatchDiversity { get; set; } = true;

    /// <summary>
    /// Stopping criterion: stop when model accuracy reaches this threshold.
    /// </summary>
    public double TargetAccuracy { get; set; } = 0.95;

    /// <summary>
    /// Stopping criterion: stop when accuracy improvement is below threshold.
    /// </summary>
    public double MinImprovementThreshold { get; set; } = 0.001;
}

/// <summary>
/// Configuration options for Continual Learning.
/// </summary>
public class ContinualLearningOptions
{
    /// <summary>
    /// Regularization strength for preventing forgetting.
    /// </summary>
    public double Lambda { get; set; } = 400.0;

    /// <summary>
    /// Memory size for replay-based methods.
    /// </summary>
    public int MemorySize { get; set; } = 1000;

    /// <summary>
    /// Memory sampling strategy.
    /// </summary>
    public MemorySamplingStrategy SamplingStrategy { get; set; } = MemorySamplingStrategy.ReservoirSampling;

    /// <summary>
    /// Whether to evaluate on all previous tasks after each task.
    /// </summary>
    public bool EvaluateAllTasks { get; set; } = true;

    /// <summary>
    /// Number of samples per task to use for importance computation.
    /// </summary>
    public int ImportanceEstimationSamples { get; set; } = 100;
}

public enum MemorySamplingStrategy
{
    ReservoirSampling,
    RingBuffer,
    PrioritizedReplay,
    ClassBalanced
}
```

### PredictionModelResult Integration

```csharp
// Add to PredictionModelResult or create new specialized result classes

/// <summary>
/// Results from an Active Learning training session.
/// </summary>
public class ActiveLearningResult<T, TInput, TOutput>
{
    /// <summary>
    /// The final trained model.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? Model { get; set; }

    /// <summary>
    /// Indices of samples selected for labeling in each iteration.
    /// </summary>
    public List<int[]> SelectedSamplesPerIteration { get; set; } = [];

    /// <summary>
    /// Model accuracy after each iteration.
    /// </summary>
    public List<double> AccuracyHistory { get; set; } = [];

    /// <summary>
    /// Total number of labeled samples used.
    /// </summary>
    public int TotalLabeledSamples { get; set; }

    /// <summary>
    /// Selection statistics from the active learning strategy.
    /// </summary>
    public Dictionary<string, T>? FinalSelectionStatistics { get; set; }
}

/// <summary>
/// Results from a Continual Learning training session.
/// </summary>
public class ContinualLearningResult<T, TInput, TOutput>
{
    /// <summary>
    /// The final trained model.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? Model { get; set; }

    /// <summary>
    /// Performance on each task after training on all tasks.
    /// </summary>
    public Dictionary<int, TaskPerformance<T>> TaskPerformances { get; set; } = [];

    /// <summary>
    /// Average accuracy across all tasks.
    /// </summary>
    public double AverageAccuracy { get; set; }

    /// <summary>
    /// Backward transfer (impact on old tasks from learning new ones).
    /// </summary>
    public double BackwardTransfer { get; set; }

    /// <summary>
    /// Forward transfer (benefit from old tasks to new ones).
    /// </summary>
    public double ForwardTransfer { get; set; }

    /// <summary>
    /// Forgetting measure for each task.
    /// </summary>
    public Dictionary<int, double> ForgettingPerTask { get; set; } = [];
}

public class TaskPerformance<T>
{
    public int TaskId { get; set; }
    public double AccuracyAfterTask { get; set; }
    public double FinalAccuracy { get; set; }
    public double Forgetting { get; set; }
}
```

## Implementation Order

### Phase 1: Active Learning Strategies (Priority Order)
1. RandomSampling (baseline)
2. MarginSampling
3. EntropySampling
4. LeastConfidenceSampling
5. CoreSetSelection
6. DensityWeightedSampling
7. InformationDensity
8. VariationRatios
9. BALD
10. BatchBALD

### Phase 2: Continual Learning Strategies (Priority Order)
1. SynapticIntelligence
2. OnlineEWC
3. AveragedGEM
4. ExperienceReplay
5. MemoryAwareSynapses
6. GenerativeReplay
7. PackNet
8. ProgressiveNeuralNetworks
9. VariationalContinualLearning

### Phase 3: Facade Integration
1. Create ActiveLearningOptions and ContinualLearningOptions
2. Add private fields to PredictionModelBuilder
3. Add ConfigureActiveLearning and ConfigureContinualLearning methods
4. Integrate into BuildAsync methods
5. Create specialized result classes

### Phase 4: Testing
1. Unit tests for each new strategy
2. Integration tests for facade
3. End-to-end tests with real models

## Summary

After implementation, AiDotNet will have:
- **15 Active Learning strategies** (5 existing + 10 new)
- **12 Continual Learning strategies** (3 existing + 9 new)
- **Full facade integration** via PredictionModelBuilder
- **Comprehensive options** for customization
- **Result tracking** for analysis and debugging

This provides exhaustive coverage of the most important AL and CL techniques from the academic literature, making AiDotNet a complete solution for these paradigms.
