using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ActiveLearning.Results;

/// <summary>
/// Result from a single active learning iteration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Each iteration of active learning involves selecting samples,
/// getting labels, and retraining the model. This class captures what happened in one iteration.</para>
/// </remarks>
public class ActiveLearningIterationResult<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    /// <param name="numOps">Optional numeric operations provider. If null, a default provider will be used.</param>
    public ActiveLearningIterationResult(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        TrainingAccuracy = _numOps.Zero;
        TrainingLoss = _numOps.Zero;
        ValidationAccuracy = _numOps.Zero;
        ValidationLoss = _numOps.Zero;
        AverageQueryScore = _numOps.Zero;
        MaxQueryScore = _numOps.Zero;
        AveragePoolUncertainty = _numOps.Zero;
    }

    /// <summary>
    /// Gets or sets the iteration number (0-indexed).
    /// </summary>
    public int IterationNumber { get; set; }

    /// <summary>
    /// Gets or sets the number of samples queried in this iteration.
    /// </summary>
    public int SamplesQueried { get; set; }

    /// <summary>
    /// Gets or sets the total number of labeled samples after this iteration.
    /// </summary>
    public int TotalLabeledSamples { get; set; }

    /// <summary>
    /// Gets or sets the training accuracy after this iteration.
    /// </summary>
    public T TrainingAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the training loss after this iteration.
    /// </summary>
    public T TrainingLoss { get; set; }

    /// <summary>
    /// Gets or sets the validation accuracy after this iteration (if available).
    /// </summary>
    public T ValidationAccuracy { get; set; }

    /// <summary>
    /// Gets or sets whether validation data was available for this iteration.
    /// </summary>
    public bool HasValidation { get; set; }

    /// <summary>
    /// Gets or sets the validation loss after this iteration (if available).
    /// </summary>
    public T ValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the average informativeness score of queried samples.
    /// </summary>
    public T AverageQueryScore { get; set; }

    /// <summary>
    /// Gets or sets the maximum informativeness score among queried samples.
    /// </summary>
    public T MaxQueryScore { get; set; }

    /// <summary>
    /// Gets or sets the indices of samples that were queried.
    /// </summary>
    public int[] QueriedIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the informativeness scores of queried samples.
    /// </summary>
    public T[] QueryScores { get; set; } = Array.Empty<T>();

    /// <summary>
    /// Gets or sets the time spent on this iteration.
    /// </summary>
    public TimeSpan IterationTime { get; set; }

    /// <summary>
    /// Gets or sets the time spent training the model.
    /// </summary>
    public TimeSpan TrainingTime { get; set; }

    /// <summary>
    /// Gets or sets the time spent selecting samples.
    /// </summary>
    public TimeSpan SelectionTime { get; set; }

    /// <summary>
    /// Gets or sets the number of unlabeled samples remaining.
    /// </summary>
    public int UnlabeledRemaining { get; set; }

    /// <summary>
    /// Gets or sets the average uncertainty of the unlabeled pool.
    /// </summary>
    public T AveragePoolUncertainty { get; set; }
}

/// <summary>
/// Final result from the complete active learning process.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class summarizes the entire active learning process,
/// including the learning curve, final performance, and efficiency metrics.</para>
/// </remarks>
public class ActiveLearningResult<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    /// <param name="numOps">Optional numeric operations provider. If null, a default provider will be used.</param>
    public ActiveLearningResult(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        BudgetUsed = _numOps.Zero;
        FinalTrainingAccuracy = _numOps.Zero;
        FinalTrainingLoss = _numOps.Zero;
        FinalValidationAccuracy = _numOps.Zero;
        FinalValidationLoss = _numOps.Zero;
        FinalTestAccuracy = _numOps.Zero;
        FinalTestLoss = _numOps.Zero;
        AreaUnderLearningCurve = _numOps.Zero;
        QueryEfficiencyRatio = _numOps.Zero;
        SampleEfficiency = _numOps.Zero;
        TargetAccuracy = _numOps.Zero;
    }

    /// <summary>
    /// Gets or sets the total number of iterations completed.
    /// </summary>
    public int TotalIterations { get; set; }

    /// <summary>
    /// Gets or sets the total number of samples labeled.
    /// </summary>
    public int TotalSamplesLabeled { get; set; }

    /// <summary>
    /// Gets or sets the initial number of labeled samples.
    /// </summary>
    public int InitialLabeledSamples { get; set; }

    /// <summary>
    /// Gets or sets the labeling budget used as a fraction of maximum.
    /// </summary>
    public T BudgetUsed { get; set; }

    /// <summary>
    /// Gets or sets the final training accuracy.
    /// </summary>
    public T FinalTrainingAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the final training loss.
    /// </summary>
    public T FinalTrainingLoss { get; set; }

    /// <summary>
    /// Gets or sets the final validation accuracy.
    /// </summary>
    public T FinalValidationAccuracy { get; set; }

    /// <summary>
    /// Gets or sets whether validation data was available.
    /// </summary>
    public bool HasValidation { get; set; }

    /// <summary>
    /// Gets or sets the final validation loss.
    /// </summary>
    public T FinalValidationLoss { get; set; }

    /// <summary>
    /// Gets or sets the final test accuracy.
    /// </summary>
    public T FinalTestAccuracy { get; set; }

    /// <summary>
    /// Gets or sets whether test data was provided.
    /// </summary>
    public bool HasTest { get; set; }

    /// <summary>
    /// Gets or sets the final test loss.
    /// </summary>
    public T FinalTestLoss { get; set; }

    /// <summary>
    /// Gets or sets the learning curve showing performance vs. labeled samples.
    /// </summary>
    public LearningCurve<T> LearningCurve { get; set; } = new();

    /// <summary>
    /// Gets or sets the results from each iteration.
    /// </summary>
    public List<ActiveLearningIterationResult<T>> IterationResults { get; set; } = new List<ActiveLearningIterationResult<T>>();

    /// <summary>
    /// Gets or sets the reason why learning stopped.
    /// </summary>
    public string StoppingReason { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the total time for the active learning process.
    /// </summary>
    public TimeSpan TotalTime { get; set; }

    /// <summary>
    /// Gets or sets the total time spent training.
    /// </summary>
    public TimeSpan TotalTrainingTime { get; set; }

    /// <summary>
    /// Gets or sets the total time spent on sample selection.
    /// </summary>
    public TimeSpan TotalSelectionTime { get; set; }

    /// <summary>
    /// Gets or sets the query strategy used.
    /// </summary>
    public string QueryStrategyName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the batch strategy used (if any).
    /// </summary>
    public string? BatchStrategyName { get; set; }

    // === Efficiency Metrics ===

    /// <summary>
    /// Gets or sets the Area Under the Learning Curve (AULC).
    /// Higher is better - measures overall learning efficiency.
    /// </summary>
    public T AreaUnderLearningCurve { get; set; }

    /// <summary>
    /// Gets or sets whether AULC was calculated.
    /// </summary>
    public bool HasAULC { get; set; }

    /// <summary>
    /// Gets or sets the query efficiency ratio.
    /// Compares performance to random sampling baseline.
    /// </summary>
    public T QueryEfficiencyRatio { get; set; }

    /// <summary>
    /// Gets or sets whether query efficiency ratio was calculated.
    /// </summary>
    public bool HasQueryEfficiencyRatio { get; set; }

    /// <summary>
    /// Gets or sets the sample efficiency (accuracy gain per sample).
    /// </summary>
    public T SampleEfficiency { get; set; }

    /// <summary>
    /// Gets or sets whether sample efficiency was calculated.
    /// </summary>
    public bool HasSampleEfficiency { get; set; }

    /// <summary>
    /// Gets or sets the number of samples needed to reach target accuracy.
    /// </summary>
    public int? SamplesToTargetAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the target accuracy used for efficiency calculations.
    /// </summary>
    public T TargetAccuracy { get; set; }

    /// <summary>
    /// Gets or sets whether target accuracy was set.
    /// </summary>
    public bool HasTargetAccuracy { get; set; }
}

/// <summary>
/// Detailed metrics for query strategy performance analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class QueryStrategyMetrics<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    /// <param name="numOps">Optional numeric operations provider. If null, a default provider will be used.</param>
    public QueryStrategyMetrics(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        AverageInformativeness = _numOps.Zero;
        InformativenessVariance = _numOps.Zero;
        AverageDiversity = _numOps.Zero;
        SelectionCorrelation = _numOps.Zero;
    }

    /// <summary>
    /// Gets or sets the name of the query strategy.
    /// </summary>
    public string StrategyName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the average informativeness score across all queries.
    /// </summary>
    public T AverageInformativeness { get; set; }

    /// <summary>
    /// Gets or sets the variance of informativeness scores.
    /// </summary>
    public T InformativenessVariance { get; set; }

    /// <summary>
    /// Gets or sets the diversity of selected samples.
    /// </summary>
    public T AverageDiversity { get; set; }

    /// <summary>
    /// Gets or sets the average time to select a batch.
    /// </summary>
    public TimeSpan AverageSelectionTime { get; set; }

    /// <summary>
    /// Gets or sets the correlation between selection order and true usefulness.
    /// </summary>
    public T SelectionCorrelation { get; set; }

    /// <summary>
    /// Gets or sets whether selection correlation was calculated.
    /// </summary>
    public bool HasSelectionCorrelation { get; set; }
}

/// <summary>
/// Comparison result between different query strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class StrategyComparisonResult<T>
{
    /// <summary>
    /// Gets or sets the strategies being compared.
    /// </summary>
    public List<string> StrategyNames { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the results for each strategy.
    /// </summary>
    public List<ActiveLearningResult<T>> Results { get; set; } = new List<ActiveLearningResult<T>>();

    /// <summary>
    /// Gets or sets the metrics for each strategy.
    /// </summary>
    public List<QueryStrategyMetrics<T>> Metrics { get; set; } = new List<QueryStrategyMetrics<T>>();

    /// <summary>
    /// Gets or sets the name of the best performing strategy.
    /// </summary>
    public string BestStrategy { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the ranking of strategies by AULC.
    /// </summary>
    public List<(string Strategy, T Score)> RankingByAULC { get; set; } = new List<(string Strategy, T Score)>();

    /// <summary>
    /// Gets or sets the ranking of strategies by final accuracy.
    /// </summary>
    public List<(string Strategy, T Score)> RankingByAccuracy { get; set; } = new List<(string Strategy, T Score)>();

    /// <summary>
    /// Gets or sets the ranking of strategies by efficiency.
    /// </summary>
    public List<(string Strategy, T Score)> RankingByEfficiency { get; set; } = new List<(string Strategy, T Score)>();
}

/// <summary>
/// Cold start analysis result showing initial sample selection performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ColdStartAnalysisResult<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    /// <param name="numOps">Optional numeric operations provider. If null, a default provider will be used.</param>
    public ColdStartAnalysisResult(INumericOperations<T>? numOps = null)
    {
        _numOps = numOps ?? MathHelper.GetNumericOperations<T>();
        InitialAccuracy = _numOps.Zero;
        Representativeness = _numOps.Zero;
    }

    /// <summary>
    /// Gets or sets the cold start strategy used.
    /// </summary>
    public string StrategyName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of initial samples selected.
    /// </summary>
    public int InitialSampleCount { get; set; }

    /// <summary>
    /// Gets or sets the accuracy after initial training.
    /// </summary>
    public T InitialAccuracy { get; set; }

    /// <summary>
    /// Gets or sets how representative the initial sample is of the full dataset.
    /// </summary>
    public T Representativeness { get; set; }

    /// <summary>
    /// Gets or sets the class balance of the initial sample.
    /// </summary>
    public Dictionary<string, T> ClassDistribution { get; set; } = new Dictionary<string, T>();

    /// <summary>
    /// Gets or sets the time to select initial samples.
    /// </summary>
    public TimeSpan SelectionTime { get; set; }
}
