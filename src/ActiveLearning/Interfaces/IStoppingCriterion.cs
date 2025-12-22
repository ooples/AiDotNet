namespace AiDotNet.ActiveLearning.Interfaces;

/// <summary>
/// Interface for stopping criteria in active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Active learning loops can be stopped based on various criteria
/// beyond just exhausting the labeling budget. Early stopping can save resources when
/// additional labels won't significantly improve the model.</para>
///
/// <para><b>Common Stopping Criteria:</b></para>
/// <list type="bullet">
/// <item><description><b>Budget Exhausted:</b> Reached maximum number of labels</description></item>
/// <item><description><b>Performance Plateau:</b> Model accuracy stopped improving</description></item>
/// <item><description><b>Confidence Threshold:</b> Model is confident on all remaining samples</description></item>
/// <item><description><b>Prediction Stability:</b> Predictions are stable across iterations</description></item>
/// </list>
/// </remarks>
public interface IStoppingCriterion<T>
{
    /// <summary>
    /// Gets the name of the stopping criterion.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets a description of when this criterion triggers.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Checks whether the stopping criterion is met.
    /// </summary>
    /// <param name="context">The current active learning context.</param>
    /// <returns>True if learning should stop, false otherwise.</returns>
    bool ShouldStop(ActiveLearningContext<T> context);

    /// <summary>
    /// Gets a progress indicator (0 to 1) showing how close to stopping.
    /// </summary>
    /// <param name="context">The current active learning context.</param>
    /// <returns>Progress value between 0 (just started) and 1 (about to stop).</returns>
    T GetProgress(ActiveLearningContext<T> context);

    /// <summary>
    /// Resets the criterion to its initial state.
    /// </summary>
    void Reset();
}

/// <summary>
/// Context information for stopping criterion evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class provides all the information a stopping criterion
/// needs to make its decision, including history of metrics and current state.</para>
/// </remarks>
public class ActiveLearningContext<T>
{
    /// <summary>
    /// Gets or sets the current iteration number.
    /// </summary>
    public int CurrentIteration { get; set; }

    /// <summary>
    /// Gets or sets the total number of labeled samples.
    /// </summary>
    public int TotalLabeled { get; set; }

    /// <summary>
    /// Gets or sets the maximum labeling budget.
    /// </summary>
    public int MaxBudget { get; set; }

    /// <summary>
    /// Gets or sets the number of unlabeled samples remaining.
    /// </summary>
    public int UnlabeledRemaining { get; set; }

    /// <summary>
    /// Gets or sets the history of training accuracy.
    /// </summary>
    public List<T> AccuracyHistory { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the history of validation accuracy.
    /// </summary>
    public List<T> ValidationAccuracyHistory { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the history of training loss.
    /// </summary>
    public List<T> LossHistory { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the history of average uncertainty scores.
    /// </summary>
    public List<T> UncertaintyHistory { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the history of query informativeness scores.
    /// </summary>
    public List<T> QueryScoreHistory { get; set; } = new List<T>();

    /// <summary>
    /// Gets or sets the predictions from the previous iteration for stability checking.
    /// </summary>
    public Vector<T>? PreviousPredictions { get; set; }

    /// <summary>
    /// Gets or sets the current predictions for stability checking.
    /// </summary>
    public Vector<T>? CurrentPredictions { get; set; }

    /// <summary>
    /// Gets or sets the time elapsed since learning started.
    /// </summary>
    public TimeSpan ElapsedTime { get; set; }

    /// <summary>
    /// Gets or sets the maximum allowed time for learning.
    /// </summary>
    public TimeSpan? MaxTime { get; set; }
}

/// <summary>
/// Interface for composite stopping criteria (multiple criteria combined).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ICompositeCriterion<T> : IStoppingCriterion<T>
{
    /// <summary>
    /// Gets the individual criteria in this composite.
    /// </summary>
    IReadOnlyList<IStoppingCriterion<T>> Criteria { get; }

    /// <summary>
    /// Adds a criterion to the composite.
    /// </summary>
    /// <param name="criterion">The criterion to add.</param>
    void AddCriterion(IStoppingCriterion<T> criterion);

    /// <summary>
    /// Removes a criterion from the composite.
    /// </summary>
    /// <param name="criterion">The criterion to remove.</param>
    /// <returns>True if the criterion was found and removed.</returns>
    bool RemoveCriterion(IStoppingCriterion<T> criterion);
}

/// <summary>
/// Interface for stopping criteria that need prediction access.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IPredictionBasedCriterion<T, TInput, TOutput> : IStoppingCriterion<T>
{
    /// <summary>
    /// Updates the criterion with new predictions.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="dataset">The dataset to make predictions on.</param>
    void UpdatePredictions(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> dataset);

    /// <summary>
    /// Computes the prediction stability metric.
    /// </summary>
    /// <returns>A value indicating how stable predictions are (higher = more stable).</returns>
    T ComputeStability();
}

/// <summary>
/// Interface for uncertainty-based stopping criteria.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IUncertaintyBasedCriterion<T> : IStoppingCriterion<T>
{
    /// <summary>
    /// Gets or sets the uncertainty threshold below which to stop.
    /// </summary>
    T UncertaintyThreshold { get; set; }

    /// <summary>
    /// Gets the current average uncertainty.
    /// </summary>
    T CurrentAverageUncertainty { get; }

    /// <summary>
    /// Gets the fraction of samples below the uncertainty threshold.
    /// </summary>
    T FractionConfident { get; }
}
