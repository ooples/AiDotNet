namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the active learning strategy to use for sample selection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Active learning strategies help identify which unlabeled samples
/// would be most valuable to label. Different strategies use different criteria to measure
/// how "informative" a sample is for training the model.</para>
/// </remarks>
public enum ActiveLearningStrategyType
{
    /// <summary>
    /// Selects samples randomly. Simple baseline strategy.
    /// </summary>
    Random,

    /// <summary>
    /// Selects samples where the model is most uncertain about predictions.
    /// Supports multiple uncertainty measures (least confidence, margin, entropy).
    /// </summary>
    UncertaintySampling,

    /// <summary>
    /// Uses multiple models and selects samples where they disagree the most.
    /// </summary>
    QueryByCommittee,

    /// <summary>
    /// Selects samples that would cause the largest change to model parameters.
    /// </summary>
    ExpectedModelChange,

    /// <summary>
    /// Selects samples that maximize coverage of the input space.
    /// </summary>
    DiversitySampling,

    /// <summary>
    /// Combines multiple criteria (uncertainty + diversity) for selection.
    /// </summary>
    HybridSampling,

    /// <summary>
    /// Selects samples based on entropy of predicted class probabilities.
    /// </summary>
    EntropySampling,

    /// <summary>
    /// Selects samples with the smallest margin between top two predictions.
    /// </summary>
    MarginSampling,

    /// <summary>
    /// Selects samples where the top prediction has the lowest confidence.
    /// </summary>
    LeastConfidenceSampling,

    /// <summary>
    /// Weights uncertainty by sample density in the input space.
    /// </summary>
    DensityWeightedSampling,

    /// <summary>
    /// Selects representative samples that form a core set of the data.
    /// </summary>
    CoreSetSelection,

    /// <summary>
    /// Combines uncertainty with representativeness based on local density.
    /// </summary>
    InformationDensity,

    /// <summary>
    /// Uses variation ratios (1 - max probability) for uncertainty estimation.
    /// </summary>
    VariationRatios,

    /// <summary>
    /// Bayesian Active Learning by Disagreement - uses mutual information between
    /// predictions and model parameters for selection.
    /// </summary>
    BALD,

    /// <summary>
    /// Batch-mode BALD that accounts for redundancy when selecting multiple samples.
    /// </summary>
    BatchBALD
}

/// <summary>
/// Represents configuration options for active learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Active learning helps when labeling data is expensive or time-consuming.
/// Instead of randomly selecting samples to label, active learning intelligently picks the samples
/// that would be most helpful for training the model. This can dramatically reduce the number of
/// labels needed while achieving similar or better performance.</para>
///
/// <para><b>Typical Usage:</b></para>
/// <code>
/// var options = new ActiveLearningOptions
/// {
///     Strategy = ActiveLearningStrategyType.UncertaintySampling,
///     BatchSize = 10,
///     UseBatchDiversity = true
/// };
/// </code>
///
/// <para><b>How to Choose a Strategy:</b></para>
/// <list type="bullet">
/// <item><description><b>UncertaintySampling:</b> Fast and effective for most classification tasks.</description></item>
/// <item><description><b>QueryByCommittee:</b> Good when you have multiple models or can train an ensemble.</description></item>
/// <item><description><b>DiversitySampling:</b> Ensures good coverage of the input space.</description></item>
/// <item><description><b>HybridSampling:</b> Balances uncertainty and diversity.</description></item>
/// <item><description><b>BALD/BatchBALD:</b> State-of-the-art for Bayesian neural networks.</description></item>
/// </list>
/// </remarks>
public class ActiveLearningOptions
{
    /// <summary>
    /// Gets or sets the active learning strategy to use.
    /// </summary>
    /// <remarks>
    /// The strategy determines how samples are selected from the unlabeled pool.
    /// Default is UncertaintySampling, which is simple and effective for most tasks.
    /// </remarks>
    public ActiveLearningStrategyType Strategy { get; set; } = ActiveLearningStrategyType.UncertaintySampling;

    /// <summary>
    /// Gets or sets the number of samples to select in each active learning iteration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many samples are labeled at once.
    /// Larger batches are more efficient but may select redundant samples.</para>
    /// <para>Typical values range from 1 (pure sequential) to 50-100 (batch mode).</para>
    /// </remarks>
    public int BatchSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to consider diversity when selecting multiple samples in a batch.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When selecting multiple samples at once, enabling batch diversity
    /// ensures the selected samples are not only informative but also different from each other.
    /// This prevents selecting redundant samples that provide similar information.</para>
    /// </remarks>
    public bool UseBatchDiversity { get; set; } = true;

    /// <summary>
    /// Gets or sets the uncertainty measure for UncertaintySampling strategy.
    /// </summary>
    /// <remarks>
    /// Options: "LeastConfidence", "MarginSampling", "Entropy".
    /// Default is "Entropy" which considers the full probability distribution.
    /// </remarks>
    public string UncertaintyMeasure { get; set; } = "Entropy";

    /// <summary>
    /// Gets or sets the number of committee members for QueryByCommittee strategy.
    /// </summary>
    /// <remarks>
    /// More committee members provide better uncertainty estimates but require more computation.
    /// Typical values range from 3 to 10.
    /// </remarks>
    public int CommitteeSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of Monte Carlo samples for BALD strategy.
    /// </summary>
    /// <remarks>
    /// More samples provide better approximation of model uncertainty but increase computation.
    /// Typical values range from 5 to 20.
    /// </remarks>
    public int NumMcSamples { get; set; } = 10;

    /// <summary>
    /// Gets or sets the dropout rate for Monte Carlo Dropout in BALD/BatchBALD.
    /// </summary>
    /// <remarks>
    /// The dropout rate controls the variation between forward passes.
    /// Typical values are 0.1 to 0.5.
    /// </remarks>
    public double DropoutRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the weight for diversity component in hybrid strategies.
    /// </summary>
    /// <remarks>
    /// Controls the balance between uncertainty (1 - DiversityWeight) and diversity (DiversityWeight).
    /// Values closer to 0 favor uncertainty, values closer to 1 favor diversity.
    /// </remarks>
    public double DiversityWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the number of clusters for diversity-based methods.
    /// </summary>
    /// <remarks>
    /// Used by CoreSetSelection and DiversitySampling to group similar samples.
    /// Setting to 0 uses the batch size as the number of clusters.
    /// </remarks>
    public int NumClusters { get; set; } = 0;

    /// <summary>
    /// Gets or sets the beta parameter for DensityWeightedSampling.
    /// </summary>
    /// <remarks>
    /// Controls the influence of density on sample selection.
    /// Higher values give more weight to samples in dense regions.
    /// </remarks>
    public double DensityBeta { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of nearest neighbors for density estimation.
    /// </summary>
    /// <remarks>
    /// Used by density-based methods to estimate local density around samples.
    /// Typical values range from 5 to 20.
    /// </remarks>
    public int NumNeighbors { get; set; } = 10;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// Setting a seed ensures the same samples are selected across runs.
    /// Leave as null for random selection each time.
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the minimum number of samples required in the unlabeled pool.
    /// </summary>
    /// <remarks>
    /// Active learning stops when the pool size falls below this threshold.
    /// </remarks>
    public int MinimumPoolSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to normalize informativeness scores before selection.
    /// </summary>
    /// <remarks>
    /// Normalization can help when combining scores from different strategies
    /// or when comparing scores across different batches.
    /// </remarks>
    public bool NormalizeScores { get; set; } = false;
}
