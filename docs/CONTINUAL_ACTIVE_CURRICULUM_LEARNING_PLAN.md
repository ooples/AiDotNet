# Continual, Active, and Curriculum Learning Implementation Plan

## Executive Summary

This document outlines a comprehensive implementation plan to bring AiDotNet's Continual Learning, Active Learning, and Curriculum Learning systems to production-ready status that **exceeds industry standards** (Avalanche, modAL, ContinuousAI benchmarks).

**Current State**: No implementations exist in the codebase.
**Target State**: 95+ files across 6 phases with comprehensive configuration, 15+ strategies per learning type, and full benchmark infrastructure.

---

## Industry Standards Analysis

### Avalanche (ContinualAI) - The Gold Standard for Continual Learning

Avalanche provides 5 core modules:
1. **Benchmarks** - CL scenarios (Class-IL, Task-IL, Domain-IL, etc.)
2. **Training** - Training loops with strategy plugins
3. **Evaluation** - Metrics (forgetting, forward/backward transfer)
4. **Logging** - TensorBoard, WandB, CSV logging
5. **Models** - Base model architectures for CL

Key strategies implemented:
- **Regularization-based**: EWC, Online-EWC, SI, LwF, MAS, LFL
- **Replay-based**: Experience Replay, GEM, A-GEM, GSS, iCARL, AGEM
- **Parameter isolation**: PackNet, Progressive Neural Networks, HAT
- **Hybrid**: BiC, LUCIR, AR1, VCL

### modAL (Active Learning) - The Python Standard

modAL provides:
- **Query Strategies**: Uncertainty sampling, QBC, expected model change, BALD
- **Batch Strategies**: Ranked batch, clustered uncertainty
- **Stopping Criteria**: Stabilizing predictions, contradicting information
- **Committee Models**: Ensemble-based disagreement

### What We Must Exceed

| Feature | Avalanche/modAL | Our Target |
|---------|-----------------|------------|
| CL Strategies | 15 | 20+ |
| AL Strategies | 8 | 15+ |
| Configuration Options | Limited | Comprehensive nullable configs |
| Type Safety | Python dynamic | C# generic types |
| GPU Support | PyTorch backend | Custom CUDA kernels |
| Benchmarks | Basic | Full statistical analysis |

---

## Phase 1: Core Infrastructure (CRITICAL)

### 1.1 Configuration System

All configuration classes use **nullable properties with industry-standard defaults**.

#### ContinualLearnerConfig.cs (Enhanced)
```csharp
namespace AiDotNet.ContinualLearning.Config;

/// <summary>
/// Comprehensive configuration for continual learning trainers.
/// All properties are nullable - null values use industry-standard defaults.
/// </summary>
public class ContinualLearnerConfig<T>
{
    // === Training Parameters ===

    /// <summary>Number of epochs per task. Default: 10</summary>
    public int? EpochsPerTask { get; set; }

    /// <summary>Batch size for training. Default: 32</summary>
    public int? BatchSize { get; set; }

    /// <summary>Learning rate. Default: 0.001</summary>
    public T? LearningRate { get; set; }

    /// <summary>Weight decay for regularization. Default: 1e-4</summary>
    public T? WeightDecay { get; set; }

    /// <summary>Momentum for SGD-based optimizers. Default: 0.9</summary>
    public T? Momentum { get; set; }

    // === Memory Parameters ===

    /// <summary>Maximum memory buffer size. Default: 5000</summary>
    public int? MemorySize { get; set; }

    /// <summary>Samples per task in memory. Default: auto-calculated</summary>
    public int? SamplesPerTask { get; set; }

    /// <summary>Memory selection strategy. Default: Reservoir</summary>
    public MemorySelectionStrategy? MemoryStrategy { get; set; }

    // === EWC-Specific Parameters ===

    /// <summary>EWC regularization strength (lambda). Default: 1000</summary>
    public T? EwcLambda { get; set; }

    /// <summary>Fisher samples for importance estimation. Default: 200</summary>
    public int? FisherSamples { get; set; }

    /// <summary>Online EWC decay factor (gamma). Default: 0.9</summary>
    public T? OnlineEwcGamma { get; set; }

    // === LwF-Specific Parameters ===

    /// <summary>Knowledge distillation temperature. Default: 2.0</summary>
    public T? DistillationTemperature { get; set; }

    /// <summary>Distillation loss weight (alpha). Default: 1.0</summary>
    public T? DistillationAlpha { get; set; }

    // === GEM-Specific Parameters ===

    /// <summary>Memory strength for gradient projection. Default: 0.5</summary>
    public T? GemMemoryStrength { get; set; }

    /// <summary>Margin for A-GEM relaxation. Default: 0.5</summary>
    public T? AgemMargin { get; set; }

    // === SI-Specific Parameters ===

    /// <summary>SI importance strength (c). Default: 1.0</summary>
    public T? SiC { get; set; }

    /// <summary>SI damping factor (xi). Default: 1e-3</summary>
    public T? SiXi { get; set; }

    // === MAS-Specific Parameters ===

    /// <summary>MAS regularization strength. Default: 1.0</summary>
    public T? MasLambda { get; set; }

    // === PackNet-Specific Parameters ===

    /// <summary>Pruning percentage per task. Default: 0.75</summary>
    public T? PrunePercent { get; set; }

    /// <summary>Post-pruning fine-tuning epochs. Default: 10</summary>
    public int? PostPruneEpochs { get; set; }

    // === Progressive NN Parameters ===

    /// <summary>Lateral connection scaling. Default: 1.0</summary>
    public T? LateralScaling { get; set; }

    // === Evaluation Parameters ===

    /// <summary>Evaluate after each epoch. Default: true</summary>
    public bool? EvaluatePerEpoch { get; set; }

    /// <summary>Log metrics to console. Default: true</summary>
    public bool? EnableLogging { get; set; }

    /// <summary>Early stopping patience. Default: 5</summary>
    public int? EarlyStoppingPatience { get; set; }

    /// <summary>Random seed for reproducibility. Default: null (random)</summary>
    public int? Seed { get; set; }

    // === Advanced Parameters ===

    /// <summary>Enable gradient clipping. Default: false</summary>
    public bool? EnableGradientClipping { get; set; }

    /// <summary>Maximum gradient norm. Default: 1.0</summary>
    public T? MaxGradNorm { get; set; }

    /// <summary>Use multi-head architecture. Default: false</summary>
    public bool? MultiHead { get; set; }

    /// <summary>Task ID embedding dimension. Default: 64</summary>
    public int? TaskEmbeddingDim { get; set; }
}

public enum MemorySelectionStrategy
{
    Reservoir,     // Random reservoir sampling
    Herding,       // Class-mean herding (iCARL)
    KCenter,       // K-center coreset selection
    GSS,           // Gradient-based sample selection
    Random,        // Simple random sampling
    Stratified     // Stratified by class
}
```

#### ActiveLearnerConfig.cs (NEW)
```csharp
namespace AiDotNet.ActiveLearning.Config;

/// <summary>
/// Comprehensive configuration for active learning.
/// All properties are nullable - null values use industry-standard defaults.
/// </summary>
public class ActiveLearnerConfig<T>
{
    // === Core Parameters ===

    /// <summary>Query batch size per iteration. Default: 10</summary>
    public int? QueryBatchSize { get; set; }

    /// <summary>Initial labeled pool size. Default: 100</summary>
    public int? InitialPoolSize { get; set; }

    /// <summary>Maximum labeling budget. Default: 1000</summary>
    public int? MaxBudget { get; set; }

    /// <summary>Training epochs per AL iteration. Default: 10</summary>
    public int? EpochsPerIteration { get; set; }

    /// <summary>Training batch size. Default: 32</summary>
    public int? TrainingBatchSize { get; set; }

    /// <summary>Learning rate. Default: 0.001</summary>
    public T? LearningRate { get; set; }

    // === Query Strategy Parameters ===

    /// <summary>Primary query strategy. Default: UncertaintySampling</summary>
    public QueryStrategyType? QueryStrategy { get; set; }

    /// <summary>Uncertainty measure for sampling. Default: Entropy</summary>
    public UncertaintyMeasure? UncertaintyMeasure { get; set; }

    // === BALD-Specific Parameters ===

    /// <summary>MC Dropout samples for BALD. Default: 20</summary>
    public int? McDropoutSamples { get; set; }

    /// <summary>Dropout rate for MC Dropout. Default: 0.5</summary>
    public T? McDropoutRate { get; set; }

    // === BatchBALD-Specific Parameters ===

    /// <summary>Number of candidates for BatchBALD. Default: 100</summary>
    public int? BatchBaldCandidates { get; set; }

    /// <summary>Use greedy approximation. Default: true</summary>
    public bool? BatchBaldGreedy { get; set; }

    // === QBC-Specific Parameters ===

    /// <summary>Committee size for QBC. Default: 5</summary>
    public int? CommitteeSize { get; set; }

    /// <summary>Committee disagreement measure. Default: VoteEntropy</summary>
    public DisagreementMeasure? DisagreementMeasure { get; set; }

    // === CoreSet-Specific Parameters ===

    /// <summary>Distance metric for coreset. Default: Euclidean</summary>
    public DistanceMetric? CoresetDistance { get; set; }

    /// <summary>Use greedy k-center. Default: true</summary>
    public bool? CoresetGreedy { get; set; }

    // === Diversity-Specific Parameters ===

    /// <summary>Diversity weight in hybrid strategies. Default: 0.5</summary>
    public T? DiversityWeight { get; set; }

    /// <summary>Clustering method for diversity. Default: KMeans</summary>
    public ClusteringMethod? DiversityClustering { get; set; }

    // === Expected Model Change Parameters ===

    /// <summary>Gradient approximation method. Default: FirstOrder</summary>
    public GradientApproximation? GradientMethod { get; set; }

    // === Stopping Criteria ===

    /// <summary>Enable automatic stopping. Default: false</summary>
    public bool? EnableAutoStop { get; set; }

    /// <summary>Stopping criterion type. Default: StabilizingPredictions</summary>
    public StoppingCriterionType? StoppingCriterion { get; set; }

    /// <summary>Patience for stopping criteria. Default: 5</summary>
    public int? StoppingPatience { get; set; }

    /// <summary>Minimum accuracy gain to continue. Default: 0.001</summary>
    public T? MinAccuracyGain { get; set; }

    // === Cold Start Parameters ===

    /// <summary>Cold start strategy. Default: Random</summary>
    public ColdStartStrategy? ColdStart { get; set; }

    /// <summary>Use stratified initial selection. Default: true</summary>
    public bool? StratifiedInitial { get; set; }

    // === Advanced Parameters ===

    /// <summary>Enable active learning with labeled noise. Default: false</summary>
    public bool? HandleLabelNoise { get; set; }

    /// <summary>Query by expected error reduction. Default: false</summary>
    public bool? ExpectedErrorReduction { get; set; }

    /// <summary>Enable warm starting between iterations. Default: true</summary>
    public bool? WarmStart { get; set; }

    /// <summary>Random seed for reproducibility. Default: null</summary>
    public int? Seed { get; set; }
}

public enum QueryStrategyType
{
    UncertaintySampling,
    BALD,
    BatchBALD,
    QBC,
    CoreSet,
    Diversity,
    Entropy,
    Margin,
    LeastConfidence,
    ExpectedModelChange,
    ExpectedErrorReduction,
    VarianceReduction,
    InformationDensity,
    Random
}

public enum UncertaintyMeasure
{
    Entropy,
    Margin,
    LeastConfidence,
    PredictiveVariance
}

public enum DisagreementMeasure
{
    VoteEntropy,
    ConsensusEntropy,
    KullbackLeiblerDivergence,
    MaxDisagreement
}

public enum DistanceMetric
{
    Euclidean,
    Cosine,
    Manhattan,
    Mahalanobis
}

public enum ClusteringMethod
{
    KMeans,
    KMedoids,
    Hierarchical,
    DBSCAN
}

public enum GradientApproximation
{
    FirstOrder,
    SecondOrder,
    FisherInformation
}

public enum StoppingCriterionType
{
    StabilizingPredictions,
    ContradictingInformation,
    BudgetExhausted,
    ConvergenceDetected,
    PerformancePlateau
}

public enum ColdStartStrategy
{
    Random,
    Stratified,
    KCenter,
    DensityBased
}
```

#### CurriculumLearnerConfig.cs (NEW)
```csharp
namespace AiDotNet.CurriculumLearning.Config;

/// <summary>
/// Comprehensive configuration for curriculum learning.
/// All properties are nullable - null values use industry-standard defaults.
/// </summary>
public class CurriculumLearnerConfig<T>
{
    // === Core Parameters ===

    /// <summary>Total training epochs. Default: 100</summary>
    public int? TotalEpochs { get; set; }

    /// <summary>Training batch size. Default: 32</summary>
    public int? BatchSize { get; set; }

    /// <summary>Learning rate. Default: 0.001</summary>
    public T? LearningRate { get; set; }

    // === Curriculum Strategy ===

    /// <summary>Curriculum strategy type. Default: SelfPacedLearning</summary>
    public CurriculumStrategyType? Strategy { get; set; }

    /// <summary>Pacing function type. Default: Linear</summary>
    public PacingFunctionType? PacingFunction { get; set; }

    // === Self-Paced Learning Parameters ===

    /// <summary>Initial threshold (lambda). Default: 0.1</summary>
    public T? InitialThreshold { get; set; }

    /// <summary>Threshold growth factor. Default: 1.1</summary>
    public T? ThresholdGrowthFactor { get; set; }

    /// <summary>Hard sample penalty (mu). Default: 0.1</summary>
    public T? HardSamplePenalty { get; set; }

    // === SPCL Parameters ===

    /// <summary>Self-paced regularization weight. Default: 0.1</summary>
    public T? SelfPacedWeight { get; set; }

    /// <summary>Curriculum regularization weight. Default: 0.1</summary>
    public T? CurriculumWeight { get; set; }

    // === Difficulty Scoring ===

    /// <summary>Difficulty scorer type. Default: LossBased</summary>
    public DifficultyScorerType? DifficultyScorer { get; set; }

    /// <summary>Use transfer learning for difficulty. Default: false</summary>
    public bool? TransferDifficulty { get; set; }

    // === Loss-Based Scorer Parameters ===

    /// <summary>Pre-compute difficulty scores. Default: true</summary>
    public bool? PrecomputeDifficulty { get; set; }

    /// <summary>Update difficulty dynamically. Default: false</summary>
    public bool? DynamicDifficulty { get; set; }

    /// <summary>Epochs for difficulty warm-up. Default: 5</summary>
    public int? DifficultyWarmupEpochs { get; set; }

    // === Model-Based Scorer Parameters ===

    /// <summary>Use ensemble for scoring. Default: false</summary>
    public bool? EnsembleScoring { get; set; }

    /// <summary>Ensemble size. Default: 5</summary>
    public int? EnsembleSize { get; set; }

    // === Pacing Function Parameters ===

    /// <summary>Initial data fraction. Default: 0.2</summary>
    public T? InitialDataFraction { get; set; }

    /// <summary>Data fraction growth rate. Default: linear increase</summary>
    public T? GrowthRate { get; set; }

    /// <summary>Epoch to reach full data. Default: 80% of total</summary>
    public int? FullDataEpoch { get; set; }

    // === Anti-Curriculum Parameters ===

    /// <summary>Enable anti-curriculum (hard first). Default: false</summary>
    public bool? AntiCurriculum { get; set; }

    /// <summary>Curriculum annealing (transition to standard). Default: false</summary>
    public bool? CurriculumAnnealing { get; set; }

    /// <summary>Annealing start epoch. Default: 50% of total</summary>
    public int? AnnealingStartEpoch { get; set; }

    // === Advanced Parameters ===

    /// <summary>Enable competence-based curriculum. Default: false</summary>
    public bool? CompetenceBased { get; set; }

    /// <summary>Competence measure. Default: ValidationAccuracy</summary>
    public CompetenceMeasure? CompetenceMeasure { get; set; }

    /// <summary>Random seed. Default: null</summary>
    public int? Seed { get; set; }
}

public enum CurriculumStrategyType
{
    SelfPacedLearning,
    SPCL,                    // Self-Paced Curriculum Learning
    TeacherStudentCurriculum,
    CompetenceBasedCurriculum,
    AutomaticCurriculum,
    ProgressiveResizing,     // For images
    ProgressiveDropout
}

public enum PacingFunctionType
{
    Linear,
    Exponential,
    Logarithmic,
    StepFunction,
    Polynomial,
    Sigmoid,
    Cosine
}

public enum DifficultyScorerType
{
    LossBased,              // Use training loss
    GradientNorm,           // Gradient magnitude
    ModelConfidence,        // Prediction confidence
    TransferScorer,         // Transfer from simpler model
    PredefinedOrder,        // Manual ordering
    DatasetStatistics,      // Based on data properties
    Ensemble                // Ensemble disagreement
}

public enum CompetenceMeasure
{
    ValidationAccuracy,
    ValidationLoss,
    TrainingLoss,
    GradientNorm,
    ParameterChange
}
```

---

## Phase 2: Continual Learning Strategies (20+ Implementations)

### Directory Structure
```
src/ContinualLearning/
├── Config/
│   └── ContinualLearnerConfig.cs
├── Interfaces/
│   ├── IContinualLearner.cs
│   ├── IContinualLearningStrategy.cs
│   ├── IMemoryBuffer.cs
│   └── IScenario.cs
├── Memory/
│   ├── ExperienceReplayBuffer.cs
│   ├── ClassBalancedBuffer.cs
│   ├── GSSBuffer.cs              # Gradient-based sample selection
│   └── HerdingBuffer.cs          # iCARL-style herding
├── Strategies/
│   ├── Regularization/
│   │   ├── ElasticWeightConsolidation.cs
│   │   ├── OnlineEWC.cs
│   │   ├── SynapticIntelligence.cs
│   │   ├── MemoryAwareSynapses.cs
│   │   ├── LearningWithoutForgetting.cs
│   │   ├── LessForget.cs
│   │   └── FunctionalRegularization.cs
│   ├── Replay/
│   │   ├── ExperienceReplay.cs
│   │   ├── GradientEpisodicMemory.cs
│   │   ├── AveragedGEM.cs
│   │   ├── iCARL.cs
│   │   ├── BiC.cs               # Bias Correction
│   │   ├── LUCIR.cs             # Large-scale IL
│   │   └── DarkExperienceReplay.cs
│   └── Isolation/
│       ├── PackNet.cs
│       ├── ProgressiveNeuralNetworks.cs
│       ├── HAT.cs               # Hard Attention to Task
│       ├── PathNet.cs
│       └── DynamicExpandableNetworks.cs
├── Trainers/
│   ├── ContinualLearnerBase.cs
│   ├── EWCTrainer.cs
│   ├── LwFTrainer.cs
│   ├── GEMTrainer.cs
│   ├── PackNetTrainer.cs
│   └── ProgressiveNNTrainer.cs
├── Scenarios/
│   ├── ClassIncrementalScenario.cs
│   ├── TaskIncrementalScenario.cs
│   ├── DomainIncrementalScenario.cs
│   └── OnlineScenario.cs
├── Metrics/
│   ├── ForgetMetrics.cs
│   ├── TransferMetrics.cs
│   ├── PlasticityMetrics.cs
│   └── StabilityPlasticityTradeoff.cs
└── Results/
    ├── ContinualLearningResult.cs
    └── TaskEvaluationResult.cs
```

### Key Strategy Implementations

Each strategy follows the pattern with thread-safe random and generic numeric operations:

```csharp
/// <summary>
/// Synaptic Intelligence (SI) - Online importance estimation.
/// Reference: Zenke et al. "Continual Learning Through Synaptic Intelligence" (2017)
/// </summary>
public class SynapticIntelligence<T, TInput, TOutput> : IContinualLearningStrategy<T, TInput, TOutput>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly ContinualLearnerConfig<T> _config;

    // SI-specific state
    private Vector<T>? _previousParameters;
    private Vector<T>? _omega;           // Importance weights
    private Vector<T>? _deltaAccumulator; // Running importance updates

    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    public SynapticIntelligence(ContinualLearnerConfig<T>? config = null)
    {
        _config = config ?? new ContinualLearnerConfig<T>();
    }

    public T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        if (_previousParameters == null || _omega == null)
            return _numOps.Zero;

        var currentParams = model.GetParameters();
        var c = _config.SiC ?? _numOps.FromDouble(1.0);
        var xi = _config.SiXi ?? _numOps.FromDouble(1e-3);

        T loss = _numOps.Zero;
        for (int i = 0; i < currentParams.Length; i++)
        {
            var diff = _numOps.Subtract(currentParams[i], _previousParameters[i]);
            var importance = _numOps.Divide(_omega[i],
                _numOps.Add(_numOps.Multiply(diff, diff), xi));
            loss = _numOps.Add(loss,
                _numOps.Multiply(_numOps.Multiply(c, importance),
                    _numOps.Multiply(diff, diff)));
        }

        return _numOps.Divide(loss, _numOps.FromDouble(2.0));
    }

    // ... other methods
}
```

---

## Phase 3: Active Learning Strategies (15+ Implementations)

### Directory Structure
```
src/ActiveLearning/
├── Config/
│   └── ActiveLearnerConfig.cs
├── Interfaces/
│   ├── IActiveLearner.cs
│   ├── IQueryStrategy.cs
│   ├── IBatchStrategy.cs
│   └── IStoppingCriterion.cs
├── Strategies/
│   ├── Uncertainty/
│   │   ├── UncertaintySampling.cs
│   │   ├── EntropySampling.cs
│   │   ├── MarginSampling.cs
│   │   └── LeastConfidenceSampling.cs
│   ├── Bayesian/
│   │   ├── BALD.cs              # Bayesian Active Learning by Disagreement
│   │   ├── BatchBALD.cs
│   │   └── VarianceReduction.cs
│   ├── Committee/
│   │   ├── QueryByCommittee.cs
│   │   ├── VoteEntropyQBC.cs
│   │   └── ConsensusEntropyQBC.cs
│   ├── Diversity/
│   │   ├── CoreSetSelection.cs
│   │   ├── KCenterGreedy.cs
│   │   └── DiversitySampling.cs
│   ├── Hybrid/
│   │   ├── BADGE.cs             # Batch Active learning by Diverse Gradient Embeddings
│   │   └── LearningLoss.cs      # Learning Loss for Active Learning
│   └── InformationBased/
│       ├── InformationDensity.cs
│       ├── ExpectedModelChange.cs
│       └── ExpectedErrorReduction.cs
├── Batch/
│   ├── RankedBatchMode.cs
│   ├── ClusteredUncertainty.cs
│   └── SubmodularSelection.cs
├── Stopping/
│   ├── StabilizingPredictions.cs
│   ├── ContradictingInformation.cs
│   ├── PerformancePlateau.cs
│   └── ConfidenceThreshold.cs
├── Core/
│   ├── ActiveLearner.cs
│   ├── OracleSimulator.cs       # For benchmarking
│   └── AnnotationPool.cs
├── Metrics/
│   ├── LearningCurve.cs
│   ├── AreaUnderLC.cs
│   └── QueryEfficiency.cs
└── Results/
    └── ActiveLearningResult.cs
```

### Key Strategy: BALD (Bayesian Active Learning by Disagreement)

```csharp
/// <summary>
/// BALD - Bayesian Active Learning by Disagreement.
/// Uses MC Dropout to estimate epistemic uncertainty for query selection.
/// Reference: Houlsby et al. "Bayesian Active Learning for Classification and Preference Learning" (2011)
/// </summary>
public class BALD<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly ActiveLearnerConfig<T> _config;

    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    public BALD(ActiveLearnerConfig<T>? config = null)
    {
        _config = config ?? new ActiveLearnerConfig<T>();
    }

    public IEnumerable<int> SelectQueries(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledPool,
        int queryCount)
    {
        var mcSamples = _config.McDropoutSamples ?? 20;
        var dropoutRate = _config.McDropoutRate ?? _numOps.FromDouble(0.5);

        var scores = new List<(int Index, T Score)>();

        for (int i = 0; i < unlabeledPool.Count; i++)
        {
            var input = unlabeledPool.GetInput(i);
            var baldi = ComputeBALDScore(model, input, mcSamples, dropoutRate);
            scores.Add((i, baldi));
        }

        // Sort by BALD score (highest mutual information first)
        return scores
            .OrderByDescending(x => Convert.ToDouble(x.Score))
            .Take(queryCount)
            .Select(x => x.Index);
    }

    private T ComputeBALDScore(
        IFullModel<T, TInput, TOutput> model,
        TInput input,
        int mcSamples,
        T dropoutRate)
    {
        // Collect MC Dropout predictions
        var predictions = new List<Vector<T>>();
        for (int s = 0; s < mcSamples; s++)
        {
            model.SetDropoutMode(true, dropoutRate);
            var pred = model.Predict(input);
            predictions.Add(ConvertToVector(pred));
        }
        model.SetDropoutMode(false, _numOps.Zero);

        // Compute mean prediction
        var meanPred = ComputeMeanPrediction(predictions);

        // H[y|x,D] - Predictive entropy
        var predictiveEntropy = ComputeEntropy(meanPred);

        // E[H[y|x,w,D]] - Expected entropy
        var expectedEntropy = _numOps.Zero;
        foreach (var pred in predictions)
        {
            expectedEntropy = _numOps.Add(expectedEntropy, ComputeEntropy(pred));
        }
        expectedEntropy = _numOps.Divide(expectedEntropy, _numOps.FromDouble(mcSamples));

        // BALD = H[y|x,D] - E[H[y|x,w,D]] (mutual information)
        return _numOps.Subtract(predictiveEntropy, expectedEntropy);
    }

    private T ComputeEntropy(Vector<T> probabilities)
    {
        T entropy = _numOps.Zero;
        for (int i = 0; i < probabilities.Length; i++)
        {
            var p = probabilities[i];
            if (Convert.ToDouble(p) > 1e-10)
            {
                var logP = _numOps.Log(p);
                entropy = _numOps.Subtract(entropy, _numOps.Multiply(p, logP));
            }
        }
        return entropy;
    }

    // ... helper methods
}
```

---

## Phase 4: Curriculum Learning (NEW - 25+ Files)

### Directory Structure
```
src/CurriculumLearning/
├── Config/
│   └── CurriculumLearnerConfig.cs
├── Interfaces/
│   ├── ICurriculumLearner.cs
│   ├── IDifficultyScorer.cs
│   └── IPacingFunction.cs
├── DifficultyScorers/
│   ├── LossBasedScorer.cs
│   ├── GradientNormScorer.cs
│   ├── ModelConfidenceScorer.cs
│   ├── TransferScorer.cs
│   ├── DataStatisticsScorer.cs
│   └── EnsembleScorer.cs
├── PacingFunctions/
│   ├── LinearPacing.cs
│   ├── ExponentialPacing.cs
│   ├── LogarithmicPacing.cs
│   ├── StepPacing.cs
│   ├── PolynomialPacing.cs
│   ├── SigmoidPacing.cs
│   └── CosinePacing.cs
├── Strategies/
│   ├── SelfPacedLearning.cs     # Core SPL
│   ├── SPCL.cs                   # Self-Paced Curriculum Learning
│   ├── TeacherStudentCurriculum.cs
│   ├── CompetenceBasedCurriculum.cs
│   ├── AutomaticCurriculumLearning.cs
│   ├── ProgressiveResizing.cs   # Image-specific
│   └── ProgressiveDropout.cs
├── Core/
│   ├── CurriculumLearner.cs
│   ├── CurriculumScheduler.cs
│   └── DifficultyCache.cs
├── Metrics/
│   ├── CurriculumEfficiency.cs
│   ├── ConvergenceSpeed.cs
│   └── DifficultyCorrelation.cs
└── Results/
    └── CurriculumLearningResult.cs
```

### Key Implementation: Self-Paced Learning (SPL)

```csharp
/// <summary>
/// Self-Paced Learning (SPL) - Learns samples from easy to hard automatically.
/// Uses a latent weight variable v ∈ [0,1] for each sample that indicates whether
/// it should be included in the current training stage.
/// Reference: Kumar et al. "Self-Paced Learning for Latent Variable Models" (2010)
/// </summary>
public class SelfPacedLearning<T, TInput, TOutput> : ICurriculumLearner<T, TInput, TOutput>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly CurriculumLearnerConfig<T> _config;
    private readonly IDifficultyScorer<T, TInput, TOutput> _difficultyScorer;
    private readonly IPacingFunction<T> _pacingFunction;

    private Vector<T>? _sampleWeights;  // v_i ∈ [0, 1]
    private T _currentThreshold;         // λ (lambda)

    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    public SelfPacedLearning(
        CurriculumLearnerConfig<T>? config = null,
        IDifficultyScorer<T, TInput, TOutput>? scorer = null,
        IPacingFunction<T>? pacing = null)
    {
        _config = config ?? new CurriculumLearnerConfig<T>();
        _difficultyScorer = scorer ?? new LossBasedScorer<T, TInput, TOutput>();
        _pacingFunction = pacing ?? new LinearPacing<T>();

        _currentThreshold = _config.InitialThreshold ?? _numOps.FromDouble(0.1);
    }

    public CurriculumLearningResult<T> Train(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IDataset<T, TInput, TOutput> trainData)
    {
        var totalEpochs = _config.TotalEpochs ?? 100;
        var batchSize = _config.BatchSize ?? 32;
        var lr = _config.LearningRate ?? _numOps.FromDouble(0.001);

        // Initialize sample weights to 0 (nothing selected initially)
        _sampleWeights = new Vector<T>(trainData.Count);
        for (int i = 0; i < trainData.Count; i++)
            _sampleWeights[i] = _numOps.Zero;

        // Pre-compute or initialize difficulty scores
        var difficultyScores = _difficultyScorer.ComputeScores(model, trainData, lossFunction);

        var lossHistory = new List<T>();
        var samplesUsedHistory = new List<int>();

        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            // Update threshold based on pacing function
            _currentThreshold = _pacingFunction.GetThreshold(epoch, totalEpochs, _config);

            // Update sample weights based on current threshold
            UpdateSampleWeights(difficultyScores);

            // Get indices of selected samples (v_i > 0)
            var selectedIndices = GetSelectedSamples();
            samplesUsedHistory.Add(selectedIndices.Count);

            if (selectedIndices.Count == 0)
                continue;

            // Shuffle selected samples
            var shuffled = selectedIndices.OrderBy(_ => ThreadRandom.Next()).ToList();

            // Train on selected samples
            T epochLoss = _numOps.Zero;
            int batchCount = 0;

            for (int batchStart = 0; batchStart < shuffled.Count; batchStart += batchSize)
            {
                var batchEnd = Math.Min(batchStart + batchSize, shuffled.Count);
                Vector<T>? batchGradients = null;
                T batchLoss = _numOps.Zero;

                for (int i = batchStart; i < batchEnd; i++)
                {
                    var idx = shuffled[i];
                    var input = trainData.GetInput(idx);
                    var target = trainData.GetOutput(idx);

                    // Weight the gradient by sample weight v_i
                    var weight = _sampleWeights[idx];
                    var grads = model.ComputeGradients(input, target, lossFunction);

                    for (int j = 0; j < grads.Length; j++)
                        grads[j] = _numOps.Multiply(grads[j], weight);

                    if (batchGradients == null)
                        batchGradients = grads;
                    else
                        AccumulateGradients(batchGradients, grads);

                    batchLoss = _numOps.Add(batchLoss, ComputeLoss(model, input, target, lossFunction));
                }

                if (batchGradients != null)
                {
                    var batchSizeT = _numOps.FromDouble(batchEnd - batchStart);
                    for (int j = 0; j < batchGradients.Length; j++)
                        batchGradients[j] = _numOps.Divide(batchGradients[j], batchSizeT);

                    model.ApplyGradients(batchGradients, lr);
                }

                epochLoss = _numOps.Add(epochLoss, batchLoss);
                batchCount++;
            }

            if (batchCount > 0)
                epochLoss = _numOps.Divide(epochLoss, _numOps.FromDouble(batchCount));

            lossHistory.Add(epochLoss);

            // Optionally update difficulty scores if dynamic
            if (_config.DynamicDifficulty ?? false)
            {
                difficultyScores = _difficultyScorer.ComputeScores(model, trainData, lossFunction);
            }

            // Grow threshold for next epoch
            var growth = _config.ThresholdGrowthFactor ?? _numOps.FromDouble(1.1);
            _currentThreshold = _numOps.Multiply(_currentThreshold, growth);
        }

        return new CurriculumLearningResult<T>(
            lossHistory: new Vector<T>(lossHistory.ToArray()),
            samplesUsedPerEpoch: samplesUsedHistory,
            finalThreshold: _currentThreshold);
    }

    /// <summary>
    /// Updates sample weights using the hard self-paced regularization scheme:
    /// v_i = 1 if loss_i < λ, else 0
    /// </summary>
    private void UpdateSampleWeights(Vector<T> difficultyScores)
    {
        for (int i = 0; i < _sampleWeights!.Length; i++)
        {
            // Hard thresholding: include sample if its difficulty is below threshold
            var difficulty = difficultyScores[i];
            _sampleWeights[i] = Convert.ToDouble(difficulty) < Convert.ToDouble(_currentThreshold)
                ? _numOps.One
                : _numOps.Zero;
        }
    }

    private List<int> GetSelectedSamples()
    {
        var selected = new List<int>();
        for (int i = 0; i < _sampleWeights!.Length; i++)
        {
            if (Convert.ToDouble(_sampleWeights[i]) > 0.5)
                selected.Add(i);
        }
        return selected;
    }

    // ... helper methods
}
```

---

## Phase 5: Benchmark Infrastructure

### Directory Structure
```
src/Benchmarks/
├── ContinualLearning/
│   ├── SplitMNIST.cs
│   ├── PermutedMNIST.cs
│   ├── RotatedMNIST.cs
│   ├── SplitCIFAR10.cs
│   ├── SplitCIFAR100.cs
│   ├── CORe50.cs
│   ├── TinyImageNet.cs
│   └── StreamScenarios.cs
├── ActiveLearning/
│   ├── BenchmarkDatasets.cs
│   ├── OracleSimulation.cs
│   └── QueryEfficiencyMetrics.cs
├── CurriculumLearning/
│   ├── NoisyMNIST.cs
│   ├── ImbalancedDatasets.cs
│   └── LongTailDistribution.cs
├── Common/
│   ├── BenchmarkRunner.cs
│   ├── StatisticalAnalysis.cs
│   ├── ConfidenceIntervals.cs
│   └── ResultsExporter.cs
└── Results/
    ├── BenchmarkResult.cs
    └── ComparisonReport.cs
```

### Benchmark Runner Pattern

```csharp
/// <summary>
/// Comprehensive benchmark runner with statistical analysis.
/// </summary>
public class BenchmarkRunner<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public BenchmarkReport<T> RunContinualLearningBenchmark(
        IContinualLearner<T, TInput, TOutput> learner,
        IScenario<T, TInput, TOutput> scenario,
        int numRuns = 5)
    {
        var results = new List<ContinualLearningResult<T>>();

        for (int run = 0; run < numRuns; run++)
        {
            var runResult = new List<ContinualLearningResult<T>>();

            foreach (var task in scenario.GetTasks())
            {
                var taskResult = learner.LearnTask(task);
                runResult.Add(taskResult);
            }

            results.AddRange(runResult);
        }

        return new BenchmarkReport<T>
        {
            // Compute statistics with confidence intervals
            AverageAccuracy = ComputeMeanWithCI(results.Select(r => r.TrainingAccuracy)),
            AverageForgetting = ComputeMeanWithCI(results.Select(r => ComputeForgetting(r))),
            ForwardTransfer = ComputeMeanWithCI(results.Select(r => ComputeForwardTransfer(r))),
            BackwardTransfer = ComputeMeanWithCI(results.Select(r => ComputeBackwardTransfer(r))),
            PlasticityStability = ComputePlasticityStabilityTradeoff(results)
        };
    }

    private StatisticWithCI<T> ComputeMeanWithCI(IEnumerable<T> values)
    {
        var list = values.ToList();
        var mean = list.Average(v => Convert.ToDouble(v));
        var stdDev = ComputeStdDev(list);
        var n = list.Count;

        // 95% confidence interval: mean ± 1.96 * (stdDev / sqrt(n))
        var ciHalfWidth = 1.96 * stdDev / Math.Sqrt(n);

        return new StatisticWithCI<T>
        {
            Mean = _numOps.FromDouble(mean),
            StdDev = _numOps.FromDouble(stdDev),
            CILower = _numOps.FromDouble(mean - ciHalfWidth),
            CIUpper = _numOps.FromDouble(mean + ciHalfWidth),
            N = n
        };
    }

    // ... metric computation methods
}
```

---

## Phase 6: Integration with PredictionModelBuilder

### Fluent API Integration

```csharp
public class PredictionModelBuilder<T, TInput, TOutput>
{
    private ContinualLearnerConfig<T>? _continualConfig;
    private ActiveLearnerConfig<T>? _activeConfig;
    private CurriculumLearnerConfig<T>? _curriculumConfig;

    /// <summary>
    /// Configure continual learning for sequential task learning.
    /// </summary>
    public PredictionModelBuilder<T, TInput, TOutput> WithContinualLearning(
        Action<ContinualLearnerConfig<T>>? configure = null)
    {
        _continualConfig = new ContinualLearnerConfig<T>();
        configure?.Invoke(_continualConfig);
        return this;
    }

    /// <summary>
    /// Configure active learning for efficient data labeling.
    /// </summary>
    public PredictionModelBuilder<T, TInput, TOutput> WithActiveLearning(
        Action<ActiveLearnerConfig<T>>? configure = null)
    {
        _activeConfig = new ActiveLearnerConfig<T>();
        configure?.Invoke(_activeConfig);
        return this;
    }

    /// <summary>
    /// Configure curriculum learning for training optimization.
    /// </summary>
    public PredictionModelBuilder<T, TInput, TOutput> WithCurriculumLearning(
        Action<CurriculumLearnerConfig<T>>? configure = null)
    {
        _curriculumConfig = new CurriculumLearnerConfig<T>();
        configure?.Invoke(_curriculumConfig);
        return this;
    }
}
```

---

## Implementation Summary

### Files to Create

| Phase | Category | Files | Priority |
|-------|----------|-------|----------|
| 1 | Config | 3 | CRITICAL |
| 2 | Continual Learning | 35 | HIGH |
| 3 | Active Learning | 25 | HIGH |
| 4 | Curriculum Learning | 20 | HIGH |
| 5 | Benchmarks | 15 | MEDIUM |
| 6 | Integration | 2 | HIGH |
| **Total** | | **100** | |

### Code Quality Requirements

1. **Thread-safe random**: All classes use `RandomHelper.CreateSecureRandom()` with `[ThreadStatic]`
2. **Generic numeric operations**: All math uses `INumericOperations<T>` pattern
3. **Nullable configs**: All config properties are nullable with defaults applied internally
4. **XML documentation**: Full documentation with `<remarks>` for beginners
5. **No hardcoded types**: Never use `double` in generic classes, always use `T`
6. **Path safety**: All file operations use path traversal protection

### Exceeding Industry Standards

| Feature | Avalanche | modAL | AiDotNet Target |
|---------|-----------|-------|-----------------|
| Strategies | 15 | 8 | 50+ |
| Config Options | ~20 | ~10 | 80+ per category |
| Type Safety | Python | Python | C# generics |
| Benchmarks | Basic | None | Statistical analysis |
| GPU Support | PyTorch | NumPy | Custom kernels |
| Documentation | Good | Basic | Excellent + beginner focus |

---

## Next Steps

1. **Immediate**: Create Phase 1 configuration classes
2. **Week 1**: Implement core CL strategies (EWC, LwF, GEM, SI)
3. **Week 2**: Implement core AL strategies (BALD, QBC, CoreSet)
4. **Week 3**: Implement Curriculum Learning (SPL, SPCL)
5. **Week 4**: Benchmark infrastructure and integration
6. **Week 5**: Testing, documentation, and polish

This plan will bring AiDotNet's learning systems to production-ready status that genuinely exceeds the capabilities of Avalanche and modAL.
