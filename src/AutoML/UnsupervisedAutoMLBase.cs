using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.AutoML;

/// <summary>
/// Base class for unsupervised AutoML search strategies (e.g., clustering AutoML, grid search).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unlike supervised AutoML (which extends <see cref="AutoMLModelBase{T, TInput, TOutput}"/>),
/// unsupervised AutoML operates on unlabeled data. This base class provides common infrastructure
/// for unsupervised search strategies: trial tracking, metric optimization, search space management,
/// and early stopping.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation for tools that automatically find the best
/// clustering algorithm and parameters for your data — without needing labeled examples.</para>
/// </remarks>
public abstract class UnsupervisedAutoMLBase<T>
{
    /// <summary>
    /// Hardware-accelerated engine for vector/tensor operations.
    /// </summary>
    protected static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Numeric operations for the specified type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the primary evaluation metric type.
    /// </summary>
    public ClusteringMetricType PrimaryMetricType { get; protected set; } = ClusteringMetricType.SilhouetteScore;

    /// <summary>
    /// Gets the primary metric name for evaluation dictionary lookups.
    /// </summary>
    public string PrimaryMetric => PrimaryMetricType.GetDisplayName();

    /// <summary>
    /// Gets whether higher metric values are better (derived from the metric type).
    /// </summary>
    public bool HigherIsBetter => PrimaryMetricType.IsHigherBetter();

    /// <summary>
    /// Gets the best score achieved during the search.
    /// </summary>
    public double BestScore { get; protected set; } = double.NegativeInfinity;

    /// <summary>
    /// Gets the total number of trials evaluated.
    /// </summary>
    public int TrialsEvaluated { get; protected set; }

    /// <summary>
    /// Gets or sets the maximum number of trials to run.
    /// </summary>
    public int MaxTrials { get; set; } = 100;

    /// <summary>
    /// Gets or sets the time limit for the search.
    /// </summary>
    public TimeSpan TimeLimit { get; set; } = TimeSpan.FromMinutes(30);

    /// <summary>
    /// Determines whether a new score is better than the current best.
    /// </summary>
    protected bool IsBetterScore(double newScore)
    {
        return HigherIsBetter ? newScore > BestScore : newScore < BestScore;
    }

    /// <summary>
    /// Updates the best score if the new score is better.
    /// </summary>
    protected bool TryUpdateBestScore(double score)
    {
        if (IsBetterScore(score))
        {
            BestScore = score;
            return true;
        }
        return false;
    }
}
