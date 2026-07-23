using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for cross-validation.
/// </summary>
/// <remarks>
/// <para>
/// Cross-validation provides robust performance estimates by training and testing on
/// different subsets of data. These options control the validation strategy.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cross-validation helps you understand how well your model will
/// perform on new, unseen data. Instead of a single train/test split, it creates multiple
/// splits and averages the results for a more reliable estimate.
/// </para>
/// </remarks>
public class CrossValidationOptions
{
    /// <summary>
    /// Cross-validation strategy to use. Default: StratifiedKFold for classification, KFold for regression.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different strategies work better for different problems:
    /// <list type="bullet">
    /// <item>StratifiedKFold: Preserves class balance, best for classification</item>
    /// <item>TimeSeriesSplit: For time-dependent data, never uses future data</item>
    /// <item>GroupKFold: When samples are grouped (e.g., multiple samples per patient)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public CrossValidationStrategy? Strategy { get; set; }

    /// <summary>
    /// Number of folds (K). Default: 5.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More folds = more reliable estimates but slower training.
    /// 5 or 10 folds are most common. Use more folds for small datasets.</para>
    /// </remarks>
    public int? NumberOfFolds { get; set; }

    /// <summary>
    /// Number of repetitions for repeated CV. Default: 10.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Repeated CV runs multiple K-fold validations with different
    /// random splits, giving even more reliable estimates.</para>
    /// </remarks>
    public int? NumberOfRepeats { get; set; }

    /// <summary>
    /// Whether to shuffle data before splitting. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Usually set to true, but set to false for time series data
    /// where order matters.</para>
    /// </remarks>
    public bool? Shuffle { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null (random).
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Test size ratio for shuffle split strategies. Default: 0.2 (20%).
    /// </summary>
    public double? TestSize { get; set; }

    /// <summary>
    /// Group column/feature index for group-aware CV. Default: null.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your samples are grouped (e.g., multiple readings per
    /// patient), specify the group identifier to ensure all samples from a group stay together.</para>
    /// </remarks>
    public int? GroupColumnIndex { get; set; }

    /// <summary>
    /// Group labels array for group-aware CV. Alternative to GroupColumnIndex.
    /// </summary>
    public int[]? GroupLabels { get; set; }

    /// <summary>
    /// Number of samples to leave out for leave-P-out CV. Default: 1.
    /// </summary>
    public int? LeavePOutSamples { get; set; }

    /// <summary>
    /// Gap between train and test sets for time series CV. Default: 0.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In financial data, you often need a gap between training
    /// and test to prevent data leakage. Set this to the embargo period.</para>
    /// </remarks>
    public int? Gap { get; set; }

    /// <summary>
    /// Maximum training size for time series CV. Default: null (no limit).
    /// </summary>
    public int? MaxTrainSize { get; set; }

    /// <summary>
    /// Minimum training size required. Default: null (auto).
    /// </summary>
    public int? MinTrainSize { get; set; }

    /// <summary>
    /// Purge period for purged K-fold (financial CV). Default: 0.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In financial ML, predictions at time t might use information
    /// from t+1 through the label. The purge removes samples around the fold boundary.</para>
    /// </remarks>
    public int? PurgePeriod { get; set; }

    /// <summary>
    /// Embargo period for combinatorial purged CV. Default: 0.
    /// </summary>
    public int? EmbargoPeriod { get; set; }

    /// <summary>
    /// Window size for sliding window CV. Default: null (auto).
    /// </summary>
    public int? WindowSize { get; set; }

    /// <summary>
    /// Step size for sliding window CV. Default: 1.
    /// </summary>
    public int? StepSize { get; set; }

    /// <summary>
    /// Whether to return trained models from each fold. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set to true if you want to inspect the models from each fold,
    /// but this uses more memory.</para>
    /// </remarks>
    public bool? ReturnTrainedModels { get; set; }

    /// <summary>
    /// Whether to return predictions from each fold. Default: true.
    /// </summary>
    public bool? ReturnPredictions { get; set; }

    /// <summary>
    /// Whether to run folds in parallel. Default: true.
    /// </summary>
    public bool? ParallelExecution { get; set; }

    /// <summary>
    /// Maximum degree of parallelism. Default: null (use all available cores).
    /// </summary>
    public int? MaxDegreeOfParallelism { get; set; }

    /// <summary>
    /// Whether to compute per-fold variance. Default: true.
    /// </summary>
    public bool? ComputeFoldVariance { get; set; }

    /// <summary>
    /// Whether to use out-of-fold predictions for evaluation. Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Out-of-fold predictions are predictions made on samples
    /// when they were in the test set, giving an unbiased estimate.</para>
    /// </remarks>
    public bool? UseOutOfFoldPredictions { get; set; }

    /// <summary>
    /// Inner CV options for nested cross-validation. Default: null.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Nested CV uses an outer loop to evaluate and an inner loop
    /// for hyperparameter tuning. This is the most unbiased way to evaluate tuned models.</para>
    /// </remarks>
    public CrossValidationOptions? InnerCVOptions { get; set; }
}
