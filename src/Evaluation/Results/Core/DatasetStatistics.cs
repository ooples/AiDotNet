using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Results.Core;

/// <summary>
/// Statistics about the dataset used for evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Contains summary information about the evaluation dataset including sample counts,
/// class distributions, feature statistics, and data quality indicators.
/// </para>
/// <para>
/// <b>For Beginners:</b> Before trusting evaluation metrics, you need to understand your data.
/// This class tells you:
/// <list type="bullet">
/// <item>How many samples were evaluated</item>
/// <item>Class distribution (for classification) - are classes balanced?</item>
/// <item>Target variable statistics (for regression) - range and spread</item>
/// <item>Missing values, outliers, and other data quality issues</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for statistics.</typeparam>
public class DatasetStatistics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Total number of samples in the evaluation set.
    /// </summary>
    public int TotalSamples { get; init; }

    /// <summary>
    /// Number of features in the dataset.
    /// </summary>
    public int NumberOfFeatures { get; init; }

    /// <summary>
    /// Number of output dimensions (1 for scalar, >1 for multi-output).
    /// </summary>
    public int NumberOfOutputs { get; init; }

    /// <summary>
    /// Whether this is a classification task.
    /// </summary>
    public bool IsClassification { get; init; }

    /// <summary>
    /// Whether this is a multi-label classification task.
    /// </summary>
    public bool IsMultiLabel { get; init; }

    /// <summary>
    /// Number of unique classes (for classification). Null for regression.
    /// </summary>
    public int? NumberOfClasses { get; init; }

    /// <summary>
    /// Class labels (for classification). Null for regression.
    /// </summary>
    public IReadOnlyList<T>? ClassLabels { get; init; }

    /// <summary>
    /// Number of samples per class. Null for regression.
    /// </summary>
    public IReadOnlyDictionary<T, int>? ClassCounts { get; init; }

    /// <summary>
    /// Proportion of samples per class. Null for regression.
    /// </summary>
    public IReadOnlyDictionary<T, double>? ClassProportions { get; init; }

    /// <summary>
    /// Class imbalance ratio (largest class / smallest class). Null for regression.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A ratio close to 1 means balanced classes.
    /// A ratio of 10 means the largest class has 10x more samples than the smallest.
    /// Imbalanced data can skew metrics like accuracy.</para>
    /// </remarks>
    public double? ClassImbalanceRatio { get; init; }

    /// <summary>
    /// Whether classes are considered imbalanced (ratio > threshold).
    /// </summary>
    public bool? IsImbalanced { get; init; }

    /// <summary>
    /// Minimum target value (for regression). Null for classification.
    /// </summary>
    public T? TargetMin { get; init; }

    /// <summary>
    /// Maximum target value (for regression). Null for classification.
    /// </summary>
    public T? TargetMax { get; init; }

    /// <summary>
    /// Mean target value (for regression). Null for classification.
    /// </summary>
    public T? TargetMean { get; init; }

    /// <summary>
    /// Standard deviation of target (for regression). Null for classification.
    /// </summary>
    public T? TargetStdDev { get; init; }

    /// <summary>
    /// Median target value (for regression). Null for classification.
    /// </summary>
    public T? TargetMedian { get; init; }

    /// <summary>
    /// Target value skewness (for regression). Null for classification.
    /// </summary>
    public double? TargetSkewness { get; init; }

    /// <summary>
    /// Target value kurtosis (for regression). Null for classification.
    /// </summary>
    public double? TargetKurtosis { get; init; }

    /// <summary>
    /// Number of samples with missing values.
    /// </summary>
    public int MissingSamplesCount { get; init; }

    /// <summary>
    /// Proportion of samples with missing values.
    /// </summary>
    public double MissingProportion => TotalSamples > 0 ? (double)MissingSamplesCount / TotalSamples : 0;

    /// <summary>
    /// Number of detected outliers.
    /// </summary>
    public int OutlierCount { get; init; }

    /// <summary>
    /// Proportion of outliers.
    /// </summary>
    public double OutlierProportion => TotalSamples > 0 ? (double)OutlierCount / TotalSamples : 0;

    /// <summary>
    /// Number of duplicate samples.
    /// </summary>
    public int DuplicateCount { get; init; }

    /// <summary>
    /// Whether the dataset appears to be a time series (ordered, potentially autocorrelated).
    /// </summary>
    public bool AppearsTimeSeries { get; init; }

    /// <summary>
    /// Feature names, if available.
    /// </summary>
    public IReadOnlyList<string>? FeatureNames { get; init; }

    /// <summary>
    /// Per-feature statistics (min, max, mean, std, missing count).
    /// </summary>
    public IReadOnlyList<FeatureStatistics<T>>? FeatureStats { get; init; }

    /// <summary>
    /// Warnings about the dataset (imbalance, outliers, etc.).
    /// </summary>
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Gets a summary string describing the dataset.
    /// </summary>
    public string GetSummary()
    {
        var lines = new List<string>
        {
            $"Samples: {TotalSamples:N0}",
            $"Features: {NumberOfFeatures}",
            $"Outputs: {NumberOfOutputs}"
        };

        if (IsClassification && NumberOfClasses.HasValue)
        {
            lines.Add($"Classes: {NumberOfClasses}");
            if (ClassImbalanceRatio.HasValue)
            {
                lines.Add($"Imbalance Ratio: {ClassImbalanceRatio:F2}");
            }
        }
        else if (!IsClassification)
        {
            if (TargetMean != null && TargetStdDev != null)
            {
                lines.Add($"Target Mean: {NumOps.ToDouble(TargetMean):F4}");
                lines.Add($"Target StdDev: {NumOps.ToDouble(TargetStdDev!):F4}");
            }
        }

        if (MissingSamplesCount > 0)
        {
            lines.Add($"Missing: {MissingSamplesCount:N0} ({MissingProportion:P1})");
        }

        if (OutlierCount > 0)
        {
            lines.Add($"Outliers: {OutlierCount:N0} ({OutlierProportion:P1})");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Statistics for a single feature.
/// </summary>
public class FeatureStatistics<T>
{
    /// <summary>
    /// Feature index.
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    /// Feature name, if available.
    /// </summary>
    public string? Name { get; init; }

    /// <summary>
    /// Minimum value.
    /// </summary>
    public T? Min { get; init; }

    /// <summary>
    /// Maximum value.
    /// </summary>
    public T? Max { get; init; }

    /// <summary>
    /// Mean value.
    /// </summary>
    public T? Mean { get; init; }

    /// <summary>
    /// Standard deviation.
    /// </summary>
    public T? StdDev { get; init; }

    /// <summary>
    /// Median value.
    /// </summary>
    public T? Median { get; init; }

    /// <summary>
    /// Number of unique values.
    /// </summary>
    public int UniqueCount { get; init; }

    /// <summary>
    /// Number of missing values.
    /// </summary>
    public int MissingCount { get; init; }

    /// <summary>
    /// Whether this feature appears to be categorical.
    /// </summary>
    public bool AppearsCategorical { get; init; }

    /// <summary>
    /// Whether this feature has constant value.
    /// </summary>
    public bool IsConstant { get; init; }

    /// <summary>
    /// Skewness of the distribution.
    /// </summary>
    public double? Skewness { get; init; }

    /// <summary>
    /// Kurtosis of the distribution.
    /// </summary>
    public double? Kurtosis { get; init; }

    /// <summary>
    /// Number of detected outliers for this feature.
    /// </summary>
    public int OutlierCount { get; init; }
}
