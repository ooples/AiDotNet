using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Config;

/// <summary>
/// Comprehensive configuration for active learning.
/// All properties are nullable - null values use industry-standard defaults.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Active learning is a machine learning paradigm where the algorithm
/// actively selects which data points should be labeled by an oracle (human expert). This is
/// particularly useful when labeling data is expensive or time-consuming.</para>
///
/// <para><b>How Active Learning Works:</b></para>
/// <list type="number">
/// <item><description>Start with a small labeled dataset and a large unlabeled pool</description></item>
/// <item><description>Train a model on the labeled data</description></item>
/// <item><description>Use a query strategy to select the most informative unlabeled samples</description></item>
/// <item><description>Request labels for these samples from an oracle</description></item>
/// <item><description>Add newly labeled samples to the training set and repeat</description></item>
/// </list>
///
/// <para><b>Key Concepts:</b></para>
/// <list type="bullet">
/// <item><description><b>Query Strategy:</b> The method used to select which samples to label next</description></item>
/// <item><description><b>Uncertainty Sampling:</b> Select samples where the model is most uncertain</description></item>
/// <item><description><b>Query By Committee:</b> Use an ensemble to find disagreement</description></item>
/// <item><description><b>Diversity Sampling:</b> Select samples that are diverse in feature space</description></item>
/// </list>
/// </remarks>
public class ActiveLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // === Core Parameters ===

    /// <summary>
    /// Number of samples to query in each active learning iteration.
    /// Default: 10.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many samples the algorithm will ask to be labeled
    /// in each round. Smaller values give more opportunities to update the model but require
    /// more iterations.</para>
    /// </remarks>
    public int? QueryBatchSize { get; set; }

    /// <summary>
    /// Number of initially labeled samples to start with.
    /// Default: 100.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The algorithm needs some labeled data to start with.
    /// This is typically a small random sample from your data.</para>
    /// </remarks>
    public int? InitialPoolSize { get; set; }

    /// <summary>
    /// Maximum total samples that can be labeled (labeling budget).
    /// Default: 1000.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how many samples can be labeled in total.
    /// Set this based on your labeling budget or time constraints.</para>
    /// </remarks>
    public int? MaxBudget { get; set; }

    /// <summary>
    /// Number of training epochs per active learning iteration.
    /// Default: 10.
    /// </summary>
    public int? EpochsPerIteration { get; set; }

    /// <summary>
    /// Batch size for training.
    /// Default: 32.
    /// </summary>
    public int? TrainingBatchSize { get; set; }

    /// <summary>
    /// Learning rate for model training.
    /// Default: 0.001.
    /// </summary>
    public T? LearningRate { get; set; }

    // === Query Strategy Parameters ===

    /// <summary>
    /// Primary query strategy to use for sample selection.
    /// Default: UncertaintySampling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The query strategy determines how the algorithm chooses
    /// which samples to label next. Different strategies work better for different problems.</para>
    /// </remarks>
    public QueryStrategyType? QueryStrategy { get; set; }

    /// <summary>
    /// Uncertainty measure for uncertainty-based sampling.
    /// Default: Entropy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using uncertainty sampling, this determines how
    /// uncertainty is measured. Entropy considers the full probability distribution,
    /// while margin and least confidence focus on the top predictions.</para>
    /// </remarks>
    public UncertaintyMeasure? UncertaintyMeasure { get; set; }

    // === BALD-Specific Parameters ===

    /// <summary>
    /// Number of Monte Carlo Dropout forward passes for BALD.
    /// Default: 20.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> BALD uses dropout at inference time multiple times
    /// to estimate uncertainty. More samples give better estimates but are slower.</para>
    /// </remarks>
    public int? McDropoutSamples { get; set; }

    /// <summary>
    /// Dropout rate for Monte Carlo Dropout.
    /// Default: 0.5.
    /// </summary>
    public T? McDropoutRate { get; set; }

    // === BatchBALD-Specific Parameters ===

    /// <summary>
    /// Number of candidates to consider for BatchBALD selection.
    /// Default: 100.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> BatchBALD considers the joint information of a batch.
    /// This limits how many samples are considered for computational efficiency.</para>
    /// </remarks>
    public int? BatchBaldCandidates { get; set; }

    /// <summary>
    /// Whether to use greedy approximation for BatchBALD.
    /// Default: true.
    /// </summary>
    public bool? BatchBaldGreedy { get; set; }

    // === Query By Committee (QBC) Parameters ===

    /// <summary>
    /// Number of models in the committee for QBC.
    /// Default: 5.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> QBC trains multiple models and queries samples where
    /// they disagree the most. More committee members give more robust disagreement
    /// estimates but require more computation.</para>
    /// </remarks>
    public int? CommitteeSize { get; set; }

    /// <summary>
    /// How to measure disagreement in the committee.
    /// Default: VoteEntropy.
    /// </summary>
    public DisagreementMeasure? DisagreementMeasure { get; set; }

    // === CoreSet-Specific Parameters ===

    /// <summary>
    /// Distance metric for coreset selection.
    /// Default: Euclidean.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> CoreSet selection finds samples that are representative
    /// of the unlabeled pool. The distance metric determines how similarity is measured.</para>
    /// </remarks>
    public DistanceMetric? CoresetDistance { get; set; }

    /// <summary>
    /// Whether to use greedy k-center algorithm.
    /// Default: true.
    /// </summary>
    public bool? CoresetGreedy { get; set; }

    // === Diversity-Specific Parameters ===

    /// <summary>
    /// Weight for diversity in hybrid uncertainty-diversity strategies.
    /// Default: 0.5 (equal weight for uncertainty and diversity).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some strategies combine uncertainty with diversity.
    /// This weight controls the balance: 0 = pure uncertainty, 1 = pure diversity.</para>
    /// </remarks>
    public T? DiversityWeight { get; set; }

    /// <summary>
    /// Clustering method for diversity-based sampling.
    /// Default: KMeans.
    /// </summary>
    public ClusteringMethod? DiversityClustering { get; set; }

    // === Expected Model Change Parameters ===

    /// <summary>
    /// Gradient approximation method for expected model change.
    /// Default: FirstOrder.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Expected Model Change selects samples that would
    /// cause the largest change to the model. This parameter controls how that
    /// change is estimated.</para>
    /// </remarks>
    public GradientApproximation? GradientMethod { get; set; }

    // === Stopping Criteria ===

    /// <summary>
    /// Whether to enable automatic stopping before budget exhaustion.
    /// Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If enabled, the algorithm may stop early if it
    /// determines that additional labels won't significantly improve the model.</para>
    /// </remarks>
    public bool? EnableAutoStop { get; set; }

    /// <summary>
    /// Type of stopping criterion to use.
    /// Default: StabilizingPredictions.
    /// </summary>
    public StoppingCriterionType? StoppingCriterion { get; set; }

    /// <summary>
    /// Patience for stopping criteria (number of iterations without improvement).
    /// Default: 5.
    /// </summary>
    public int? StoppingPatience { get; set; }

    /// <summary>
    /// Minimum accuracy gain required to continue.
    /// Default: 0.001 (0.1% improvement).
    /// </summary>
    public T? MinAccuracyGain { get; set; }

    // === Cold Start Parameters ===

    /// <summary>
    /// Strategy for selecting initial labeled samples.
    /// Default: Random.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines how the initial labeled pool is selected.
    /// Random is simple, but stratified ensures class balance.</para>
    /// </remarks>
    public ColdStartStrategy? ColdStart { get; set; }

    /// <summary>
    /// Whether to use stratified sampling for initial selection.
    /// Default: true.
    /// </summary>
    public bool? StratifiedInitial { get; set; }

    // === Advanced Parameters ===

    /// <summary>
    /// Whether to handle label noise in the oracle's responses.
    /// Default: false.
    /// </summary>
    public bool? HandleLabelNoise { get; set; }

    /// <summary>
    /// Use expected error reduction for query selection.
    /// Default: false.
    /// </summary>
    public bool? ExpectedErrorReduction { get; set; }

    /// <summary>
    /// Enable warm starting between iterations (keep model weights).
    /// Default: true.
    /// </summary>
    public bool? WarmStart { get; set; }

    /// <summary>
    /// Random seed for reproducibility.
    /// Default: null (random).
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Whether to evaluate on a held-out test set after each iteration.
    /// Default: true.
    /// </summary>
    public bool? EvaluatePerIteration { get; set; }

    /// <summary>
    /// Fraction of data to use as held-out test set.
    /// Default: 0.2 (20%).
    /// </summary>
    public T? TestSetFraction { get; set; }

    // === Helper Methods ===

    /// <summary>
    /// Gets the effective query batch size with default fallback.
    /// </summary>
    public int GetEffectiveQueryBatchSize() => QueryBatchSize ?? 10;

    /// <summary>
    /// Gets the effective initial pool size with default fallback.
    /// </summary>
    public int GetEffectiveInitialPoolSize() => InitialPoolSize ?? 100;

    /// <summary>
    /// Gets the effective max budget with default fallback.
    /// </summary>
    public int GetEffectiveMaxBudget() => MaxBudget ?? 1000;

    /// <summary>
    /// Gets the effective learning rate with default fallback.
    /// </summary>
    public T GetEffectiveLearningRate() => LearningRate ?? NumOps.FromDouble(0.001);

    /// <summary>
    /// Gets the effective query strategy with default fallback.
    /// </summary>
    public QueryStrategyType GetEffectiveQueryStrategy() =>
        QueryStrategy ?? QueryStrategyType.UncertaintySampling;
}

/// <summary>
/// Types of query strategies for active learning.
/// </summary>
public enum QueryStrategyType
{
    /// <summary>Query samples where the model is most uncertain.</summary>
    UncertaintySampling,

    /// <summary>Bayesian Active Learning by Disagreement - uses MC Dropout.</summary>
    BALD,

    /// <summary>Batch variant of BALD that considers joint information.</summary>
    BatchBALD,

    /// <summary>Query By Committee - uses ensemble disagreement.</summary>
    QBC,

    /// <summary>Select samples that are representative of the unlabeled pool.</summary>
    CoreSet,

    /// <summary>Select diverse samples in feature space.</summary>
    Diversity,

    /// <summary>Select samples with highest prediction entropy.</summary>
    Entropy,

    /// <summary>Select samples with smallest margin between top predictions.</summary>
    Margin,

    /// <summary>Select samples with lowest maximum predicted probability.</summary>
    LeastConfidence,

    /// <summary>Select samples that would cause largest gradient update.</summary>
    ExpectedModelChange,

    /// <summary>Select samples that would reduce expected error most.</summary>
    ExpectedErrorReduction,

    /// <summary>Select samples that would reduce prediction variance most.</summary>
    VarianceReduction,

    /// <summary>Combine uncertainty with density in feature space.</summary>
    InformationDensity,

    /// <summary>Random sampling baseline.</summary>
    Random,

    /// <summary>BADGE: Batch Active learning by Diverse Gradient Embeddings.</summary>
    BADGE,

    /// <summary>Learning Loss - learns to predict loss for query selection.</summary>
    LearningLoss
}

/// <summary>
/// Methods for measuring uncertainty in predictions.
/// </summary>
public enum UncertaintyMeasure
{
    /// <summary>Shannon entropy of the predicted probability distribution.</summary>
    Entropy,

    /// <summary>Difference between top two predicted probabilities.</summary>
    Margin,

    /// <summary>One minus the maximum predicted probability.</summary>
    LeastConfidence,

    /// <summary>Variance of predictions under MC Dropout.</summary>
    PredictiveVariance
}

/// <summary>
/// Methods for measuring disagreement in a committee of models.
/// </summary>
public enum DisagreementMeasure
{
    /// <summary>Entropy of the vote distribution across committee members.</summary>
    VoteEntropy,

    /// <summary>Entropy of the averaged probability predictions.</summary>
    ConsensusEntropy,

    /// <summary>Average KL divergence between individual and consensus predictions.</summary>
    KullbackLeiblerDivergence,

    /// <summary>Maximum disagreement between any two committee members.</summary>
    MaxDisagreement
}

/// <summary>
/// Distance metrics for diversity-based sampling.
/// </summary>
public enum DistanceMetric
{
    /// <summary>Standard Euclidean (L2) distance.</summary>
    Euclidean,

    /// <summary>Cosine similarity (1 - cos(a,b)).</summary>
    Cosine,

    /// <summary>Manhattan (L1) distance.</summary>
    Manhattan,

    /// <summary>Mahalanobis distance accounting for covariance.</summary>
    Mahalanobis
}

/// <summary>
/// Clustering methods for diversity-based sampling.
/// </summary>
public enum ClusteringMethod
{
    /// <summary>K-Means clustering.</summary>
    KMeans,

    /// <summary>K-Medoids clustering (more robust to outliers).</summary>
    KMedoids,

    /// <summary>Hierarchical agglomerative clustering.</summary>
    Hierarchical,

    /// <summary>DBSCAN density-based clustering.</summary>
    DBSCAN
}

/// <summary>
/// Methods for approximating gradient-based importance.
/// </summary>
public enum GradientApproximation
{
    /// <summary>First-order gradient approximation.</summary>
    FirstOrder,

    /// <summary>Second-order (Hessian) approximation.</summary>
    SecondOrder,

    /// <summary>Fisher Information approximation.</summary>
    FisherInformation
}

/// <summary>
/// Criteria for early stopping in active learning.
/// </summary>
public enum StoppingCriterionType
{
    /// <summary>Stop when predictions stabilize across iterations.</summary>
    StabilizingPredictions,

    /// <summary>Stop when new labels contradict previous learning.</summary>
    ContradictingInformation,

    /// <summary>Stop when labeling budget is exhausted.</summary>
    BudgetExhausted,

    /// <summary>Stop when model performance converges.</summary>
    ConvergenceDetected,

    /// <summary>Stop when accuracy improvement plateaus.</summary>
    PerformancePlateau
}

/// <summary>
/// Strategies for cold-start sample selection.
/// </summary>
public enum ColdStartStrategy
{
    /// <summary>Random sample selection.</summary>
    Random,

    /// <summary>Stratified sampling to ensure class balance.</summary>
    Stratified,

    /// <summary>K-Center greedy selection for diversity.</summary>
    KCenter,

    /// <summary>Density-based selection from high-density regions.</summary>
    DensityBased
}
