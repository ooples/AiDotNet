using AiDotNet.Evaluation.Enums;

namespace AiDotNet.Evaluation.Options;

/// <summary>
/// Configuration options for validation curve analysis.
/// </summary>
/// <remarks>
/// <para>
/// Validation curves show how model performance changes as a hyperparameter varies.
/// This helps find optimal hyperparameter values and diagnose overfitting.
/// </para>
/// <para>
/// <b>For Beginners:</b> A validation curve plots model performance (y-axis) against a
/// hyperparameter value (x-axis). For example, plotting accuracy vs. regularization strength
/// helps you find the best regularization value. The point where train and validation scores
/// are both high and close together is often optimal.
/// </para>
/// </remarks>
public class ValidationCurveOptions
{
    /// <summary>
    /// Name of the parameter to vary. Required.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the name of the hyperparameter you want to tune,
    /// like "LearningRate", "RegularizationStrength", or "NumberOfTrees".</para>
    /// </remarks>
    public string? ParameterName { get; set; }

    /// <summary>
    /// Values to test for the parameter. Required.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A list of values to try. For example, for learning rate:
    /// [0.001, 0.01, 0.1, 1.0]. Choose a range that spans from too small to too large.</para>
    /// </remarks>
    public double[]? ParameterValues { get; set; }

    /// <summary>
    /// Whether to use logarithmic spacing for parameter values. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use log spacing for parameters like learning rate or
    /// regularization where values span multiple orders of magnitude.</para>
    /// </remarks>
    public bool? UseLogScale { get; set; }

    /// <summary>
    /// Cross-validation strategy. Default: StratifiedKFold for classification.
    /// </summary>
    public CrossValidationStrategy? CVStrategy { get; set; }

    /// <summary>
    /// Number of CV folds. Default: 5.
    /// </summary>
    public int? CVFolds { get; set; }

    /// <summary>
    /// Whether to shuffle before CV. Default: true.
    /// </summary>
    public bool? Shuffle { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Metrics to track. Default: primary metric for task type.
    /// </summary>
    public IReadOnlyList<string>? MetricsToTrack { get; set; }

    /// <summary>
    /// Whether to compute confidence intervals. Default: true.
    /// </summary>
    public bool? ComputeConfidenceIntervals { get; set; }

    /// <summary>
    /// Confidence level. Default: 0.95.
    /// </summary>
    public double? ConfidenceLevel { get; set; }

    /// <summary>
    /// Whether to run in parallel. Default: true.
    /// </summary>
    public bool? ParallelExecution { get; set; }

    /// <summary>
    /// Maximum parallelism. Default: null (all cores).
    /// </summary>
    public int? MaxDegreeOfParallelism { get; set; }

    /// <summary>
    /// Whether to find optimal parameter value. Default: true.
    /// </summary>
    public bool? FindOptimalValue { get; set; }

    /// <summary>
    /// How to select optimal value. Default: MaxValidationScore.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Options are:
    /// <list type="bullet">
    /// <item>MaxValidationScore: Pick value with highest validation score</item>
    /// <item>OneSERule: Pick simplest value within 1 std error of max (more conservative)</item>
    /// <item>ElbowMethod: Find the "elbow" where improvement slows</item>
    /// </list>
    /// </para>
    /// </remarks>
    public OptimalValueSelectionMethod? OptimalValueMethod { get; set; }

    /// <summary>
    /// Whether higher parameter values mean more model complexity. Default: null (auto-detect).
    /// </summary>
    public bool? HigherValueMoreComplex { get; set; }

    /// <summary>
    /// Custom parameter setter function name. Default: null (use property reflection).
    /// </summary>
    public string? ParameterSetterMethod { get; set; }
}

/// <summary>
/// Method for selecting optimal hyperparameter value from validation curve.
/// </summary>
public enum OptimalValueSelectionMethod
{
    /// <summary>
    /// Select value with highest validation score.
    /// </summary>
    MaxValidationScore = 0,

    /// <summary>
    /// One standard error rule: select simplest value within 1 SE of maximum.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a more conservative choice that prefers simpler
    /// models when they perform nearly as well as more complex ones.</para>
    /// </remarks>
    OneSERule = 1,

    /// <summary>
    /// Elbow method: find point where improvement diminishes.
    /// </summary>
    ElbowMethod = 2,

    /// <summary>
    /// Select value where train and validation scores are closest.
    /// </summary>
    MinTrainTestGap = 3
}
