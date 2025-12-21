using AiDotNet.CurriculumLearning;
using AiDotNet.CurriculumLearning.Interfaces;
using AiDotNet.CurriculumLearning.Schedulers;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for Curriculum Learning through the AiDotNet facade.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The model input type.</typeparam>
/// <typeparam name="TOutput">The model output type.</typeparam>
/// <remarks>
/// <para>
/// This options class is designed for use with <c>PredictionModelBuilder</c>.
/// It follows the AiDotNet facade pattern: users provide minimal configuration, and the library supplies
/// industry-standard defaults internally.
/// </para>
/// <para>
/// <b>For Beginners:</b> Curriculum Learning trains models by presenting samples in order of difficulty,
/// starting with easy examples and gradually introducing harder ones. This often leads to faster
/// convergence and better final performance compared to random training order.
/// </para>
/// <para>
/// <b>Key Concepts:</b>
/// </para>
/// <list type="bullet">
/// <item><description><b>Difficulty Estimation:</b> How the system determines which samples are easy vs hard</description></item>
/// <item><description><b>Scheduling:</b> How quickly to progress from easy to hard samples</description></item>
/// <item><description><b>Phases:</b> Discrete curriculum stages with increasing difficulty</description></item>
/// </list>
/// </remarks>
public class CurriculumLearningOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the curriculum schedule type that controls progression from easy to hard samples.
    /// </summary>
    /// <remarks>
    /// <para>Default is <see cref="CurriculumScheduleType.Linear"/> which provides smooth progression.</para>
    /// <para><b>Available Strategies:</b></para>
    /// <list type="bullet">
    /// <item><description><b>Linear:</b> Steady progression (recommended for most cases)</description></item>
    /// <item><description><b>Exponential:</b> Slower start, faster finish</description></item>
    /// <item><description><b>Step:</b> Discrete jumps between difficulty levels</description></item>
    /// <item><description><b>SelfPaced:</b> Model determines its own pace based on current loss</description></item>
    /// <item><description><b>CompetenceBased:</b> Advances when model demonstrates mastery</description></item>
    /// <item><description><b>Polynomial:</b> Polynomial curve progression</description></item>
    /// <item><description><b>Cosine:</b> Cosine annealing progression</description></item>
    /// </list>
    /// </remarks>
    public CurriculumScheduleType ScheduleType { get; set; } = CurriculumScheduleType.Linear;

    /// <summary>
    /// Gets or sets the difficulty estimator type that determines sample difficulty.
    /// </summary>
    /// <remarks>
    /// <para>If null, the library auto-selects based on task type and available information.</para>
    /// </remarks>
    public DifficultyEstimatorType? DifficultyEstimator { get; set; }

    /// <summary>
    /// Gets or sets the number of curriculum phases.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 5 phases which works well for most scenarios.</para>
    /// <para>More phases = finer-grained difficulty progression.</para>
    /// </remarks>
    public int? NumPhases { get; set; }

    /// <summary>
    /// Gets or sets the total number of training epochs.
    /// </summary>
    /// <remarks>
    /// <para>If null, uses the epochs configured in the main training settings.</para>
    /// </remarks>
    public int? TotalEpochs { get; set; }

    /// <summary>
    /// Gets or sets the initial fraction of data to use (easiest samples).
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 0.2 (20% of easiest samples).</para>
    /// <para>Should be between 0.0 and 1.0.</para>
    /// </remarks>
    public double? InitialDataFraction { get; set; }

    /// <summary>
    /// Gets or sets the final fraction of data to use (typically 1.0 for all samples).
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 1.0 (all samples included by end of training).</para>
    /// <para>Should be between InitialDataFraction and 1.0.</para>
    /// </remarks>
    public double? FinalDataFraction { get; set; }

    /// <summary>
    /// Gets or sets early stopping options for curriculum learning.
    /// </summary>
    /// <remarks>
    /// <para>If null, early stopping is enabled with default patience of 10 epochs.</para>
    /// </remarks>
    public CurriculumEarlyStoppingOptions? EarlyStopping { get; set; }

    /// <summary>
    /// Gets or sets self-paced learning options.
    /// </summary>
    /// <remarks>
    /// <para>Only used when <see cref="ScheduleType"/> is <see cref="CurriculumScheduleType.SelfPaced"/>.</para>
    /// <para>If null, sensible defaults are used.</para>
    /// </remarks>
    public SelfPacedOptions? SelfPaced { get; set; }

    /// <summary>
    /// Gets or sets competence-based learning options.
    /// </summary>
    /// <remarks>
    /// <para>Only used when <see cref="ScheduleType"/> is <see cref="CurriculumScheduleType.CompetenceBased"/>.</para>
    /// <para>If null, sensible defaults are used.</para>
    /// </remarks>
    public CompetenceBasedOptions? CompetenceBased { get; set; }

    /// <summary>
    /// Gets or sets whether to recalculate sample difficulties during training.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to false (difficulties are computed once at the start).</para>
    /// <para>Setting to true enables dynamic curriculum that adapts as the model learns.</para>
    /// </remarks>
    public bool? RecalculateDifficulties { get; set; }

    /// <summary>
    /// Gets or sets how often to recalculate difficulties (in epochs).
    /// </summary>
    /// <remarks>
    /// <para>Only used when <see cref="RecalculateDifficulties"/> is true.</para>
    /// <para>If null, defaults to every 10 epochs.</para>
    /// </remarks>
    public int? DifficultyRecalculationFrequency { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize difficulty scores to [0, 1].
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to true.</para>
    /// </remarks>
    public bool? NormalizeDifficulties { get; set; }

    /// <summary>
    /// Gets or sets whether to shuffle samples within each curriculum phase.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to true for better generalization.</para>
    /// </remarks>
    public bool? ShuffleWithinPhase { get; set; }

    /// <summary>
    /// Gets or sets whether to weight sample contributions by difficulty.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to false.</para>
    /// <para>When enabled, harder samples contribute more to the loss.</para>
    /// </remarks>
    public bool? UseDifficultyWeighting { get; set; }

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <remarks>
    /// <para>If null, uses the batch size from main training settings or defaults to 32.</para>
    /// </remarks>
    public int? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets a random seed for reproducibility.
    /// </summary>
    /// <remarks>
    /// <para>If null, training is non-deterministic.</para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the verbosity level for logging.
    /// </summary>
    public CurriculumVerbosity Verbosity { get; set; } = CurriculumVerbosity.Normal;
}

/// <summary>
/// Early stopping options for curriculum learning.
/// </summary>
public class CurriculumEarlyStoppingOptions
{
    /// <summary>
    /// Gets or sets whether early stopping is enabled.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to true.</para>
    /// </remarks>
    public bool? Enabled { get; set; }

    /// <summary>
    /// Gets or sets the patience (epochs without improvement before stopping).
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 10 epochs.</para>
    /// </remarks>
    public int? Patience { get; set; }

    /// <summary>
    /// Gets or sets the minimum improvement required to reset patience counter.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 0.001.</para>
    /// </remarks>
    public double? MinDelta { get; set; }
}

/// <summary>
/// Options specific to self-paced curriculum learning.
/// </summary>
public class SelfPacedOptions
{
    /// <summary>
    /// Gets or sets the initial pace threshold (lambda).
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 0.1.</para>
    /// <para>Samples with loss below this threshold are included in training.</para>
    /// </remarks>
    public double? InitialLambda { get; set; }

    /// <summary>
    /// Gets or sets the maximum pace threshold.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 10.0.</para>
    /// </remarks>
    public double? MaxLambda { get; set; }

    /// <summary>
    /// Gets or sets how much to increase lambda each epoch.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 0.1.</para>
    /// </remarks>
    public double? LambdaGrowthRate { get; set; }

    /// <summary>
    /// Gets or sets the self-pace regularizer type.
    /// </summary>
    public SelfPaceRegularizer Regularizer { get; set; } = SelfPaceRegularizer.Hard;
}

/// <summary>
/// Options specific to competence-based curriculum learning.
/// </summary>
public class CompetenceBasedOptions
{
    /// <summary>
    /// Gets or sets the competence threshold required to advance phases.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 0.9 (90% competence required).</para>
    /// <para>Should be between 0.0 and 1.0.</para>
    /// </remarks>
    public double? CompetenceThreshold { get; set; }

    /// <summary>
    /// Gets or sets the type of competence metric to use.
    /// </summary>
    public CompetenceMetricType MetricType { get; set; } = CompetenceMetricType.Combined;

    /// <summary>
    /// Gets or sets the patience epochs for plateau detection.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 5 epochs.</para>
    /// </remarks>
    public int? PatienceEpochs { get; set; }

    /// <summary>
    /// Gets or sets the minimum improvement to reset patience.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 0.001.</para>
    /// </remarks>
    public double? MinImprovement { get; set; }

    /// <summary>
    /// Gets or sets the exponential smoothing factor for competence updates.
    /// </summary>
    /// <remarks>
    /// <para>If null, defaults to 0.3.</para>
    /// </remarks>
    public double? SmoothingFactor { get; set; }
}

/// <summary>
/// Types of difficulty estimators for curriculum learning.
/// </summary>
public enum DifficultyEstimatorType
{
    /// <summary>
    /// Uses model loss on samples to estimate difficulty.
    /// </summary>
    LossBased,

    /// <summary>
    /// Uses gradient magnitudes to estimate difficulty.
    /// </summary>
    GradientBased,

    /// <summary>
    /// Uses prediction confidence to estimate difficulty.
    /// </summary>
    ConfidenceBased,

    /// <summary>
    /// Uses sample complexity metrics (feature variance, etc.).
    /// </summary>
    ComplexityBased,

    /// <summary>
    /// Combines multiple estimators for robust difficulty estimation.
    /// </summary>
    Ensemble
}
