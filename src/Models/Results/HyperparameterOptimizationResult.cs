using AiDotNet.Interfaces;

namespace AiDotNet.Models.Results;

/// <summary>
/// Contains the results of a hyperparameter optimization process.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This stores everything about a hyperparameter search,
/// including the best hyperparameters found and all the trials that were run.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class HyperparameterOptimizationResult<T>
{
    /// <summary>
    /// Gets or sets the best trial found during optimization.
    /// </summary>
    public HyperparameterTrial<T>? BestTrial { get; set; }

    /// <summary>
    /// Gets or sets all trials performed.
    /// </summary>
    public List<HyperparameterTrial<T>> AllTrials { get; set; }

    /// <summary>
    /// Gets or sets the best hyperparameter values.
    /// </summary>
    public Dictionary<string, object> BestParameters { get; set; }

    /// <summary>
    /// Gets or sets the best objective value achieved.
    /// </summary>
    public T? BestObjectiveValue { get; set; }

    /// <summary>
    /// Gets or sets the search space that was used.
    /// </summary>
    public HyperparameterSearchSpace SearchSpace { get; set; }

    /// <summary>
    /// Gets or sets the total number of trials run.
    /// </summary>
    public int TotalTrials { get; set; }

    /// <summary>
    /// Gets or sets the number of completed trials.
    /// </summary>
    public int CompletedTrials { get; set; }

    /// <summary>
    /// Gets or sets the number of pruned trials.
    /// </summary>
    public int PrunedTrials { get; set; }

    /// <summary>
    /// Gets or sets the number of failed trials.
    /// </summary>
    public int FailedTrials { get; set; }

    /// <summary>
    /// Gets or sets the total optimization time.
    /// </summary>
    public TimeSpan TotalTime { get; set; }

    /// <summary>
    /// Gets or sets the start time of optimization.
    /// </summary>
    public DateTime StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time of optimization.
    /// </summary>
    public DateTime EndTime { get; set; }

    /// <summary>
    /// Initializes a new instance of the HyperparameterOptimizationResult class.
    /// </summary>
    public HyperparameterOptimizationResult()
    {
        AllTrials = new List<HyperparameterTrial<T>>();
        BestParameters = new Dictionary<string, object>();
        SearchSpace = new HyperparameterSearchSpace();
    }

    /// <summary>
    /// Gets the optimization history as a list of (trial number, objective value) pairs.
    /// </summary>
    public List<(int TrialNumber, T ObjectiveValue)> GetOptimizationHistory()
    {
        return AllTrials
            .Where(t => t.Status == TrialStatus.Complete && t.ObjectiveValue != null)
            .Select(t => (t.TrialNumber, t.ObjectiveValue!))
            .OrderBy(t => t.TrialNumber)
            .ToList();
    }

    /// <summary>
    /// Gets the top N trials by objective value.
    /// </summary>
    public List<HyperparameterTrial<T>> GetTopTrials(int n, bool maximize = true)
    {
        var completed = AllTrials
            .Where(t => t.Status == TrialStatus.Complete && t.ObjectiveValue != null);

        return (maximize
            ? completed.OrderByDescending(t => t.ObjectiveValue)
            : completed.OrderBy(t => t.ObjectiveValue))
            .Take(n)
            .ToList();
    }
}
