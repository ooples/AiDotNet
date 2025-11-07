using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Implements random search hyperparameter optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Random search randomly tries different hyperparameter combinations.
/// While simple, it's surprisingly effective and often outperforms grid search, especially
/// when some hyperparameters are more important than others.
///
/// How it works:
/// 1. Randomly sample hyperparameter values from the search space
/// 2. Train/evaluate the model with those hyperparameters
/// 3. Record the results
/// 4. Repeat for the specified number of trials
/// 5. Return the best configuration found
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class RandomSearchOptimizer<T, TInput, TOutput> : IHyperparameterOptimizer<T, TInput, TOutput>
{
    private readonly List<HyperparameterTrial<T>> _trials;
    private readonly Random _random;
    private readonly bool _maximize;
    private HyperparameterSearchSpace? _searchSpace;

    /// <summary>
    /// Initializes a new instance of the RandomSearchOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses a random seed.</param>
    public RandomSearchOptimizer(bool maximize = true, int? seed = null)
    {
        _trials = new List<HyperparameterTrial<T>>();
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _maximize = maximize;
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration.
    /// </summary>
    public HyperparameterOptimizationResult<T> Optimize(
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        int nTrials)
    {
        if (objectiveFunction == null)
            throw new ArgumentNullException(nameof(objectiveFunction));

        if (searchSpace == null)
            throw new ArgumentNullException(nameof(searchSpace));

        if (nTrials <= 0)
            throw new ArgumentException("Number of trials must be positive.", nameof(nTrials));

        _searchSpace = searchSpace;
        var startTime = DateTime.UtcNow;

        for (int i = 0; i < nTrials; i++)
        {
            var trial = new HyperparameterTrial<T>(i);

            // Sample hyperparameters
            var parameters = SuggestNext(trial);
            trial.Parameters = parameters;

            try
            {
                // Evaluate objective function
                var objectiveValue = objectiveFunction(parameters);
                trial.Complete(objectiveValue);
            }
            catch (Exception ex)
            {
                trial.Fail();
                trial.UserAttributes["error"] = ex.Message;
            }

            _trials.Add(trial);
        }

        var endTime = DateTime.UtcNow;

        // Find best trial
        var completedTrials = _trials.Where(t => t.Status == TrialStatus.Complete).ToList();
        if (completedTrials.Count == 0)
            throw new InvalidOperationException("No trials completed successfully.");

        var bestTrial = _maximize
            ? completedTrials.OrderByDescending(t => t.ObjectiveValue).First()
            : completedTrials.OrderBy(t => t.ObjectiveValue).First();

        return new HyperparameterOptimizationResult<T>
        {
            BestTrial = bestTrial,
            AllTrials = new List<HyperparameterTrial<T>>(_trials),
            BestParameters = bestTrial.Parameters,
            BestObjectiveValue = bestTrial.ObjectiveValue!,
            SearchSpace = searchSpace,
            TotalTrials = nTrials,
            CompletedTrials = _trials.Count(t => t.Status == TrialStatus.Complete),
            PrunedTrials = _trials.Count(t => t.Status == TrialStatus.Pruned),
            FailedTrials = _trials.Count(t => t.Status == TrialStatus.Failed),
            TotalTime = endTime - startTime,
            StartTime = startTime,
            EndTime = endTime
        };
    }

    /// <summary>
    /// Searches for the best hyperparameters for a specific model.
    /// </summary>
    public HyperparameterOptimizationResult<T> OptimizeModel(
        IModel<TInput, TOutput, TMetadata> model,
        (TInput X, TOutput Y) trainingData,
        (TInput X, TOutput Y) validationData,
        HyperparameterSearchSpace searchSpace,
        int nTrials) where TMetadata : class
    {
        throw new NotImplementedException("Model-specific optimization requires custom objective function. Use Optimize() method instead.");
    }

    /// <summary>
    /// Gets the best trial from the optimization.
    /// </summary>
    public HyperparameterTrial<T> GetBestTrial()
    {
        var completedTrials = _trials.Where(t => t.Status == TrialStatus.Complete).ToList();
        if (completedTrials.Count == 0)
            throw new InvalidOperationException("No completed trials available.");

        return _maximize
            ? completedTrials.OrderByDescending(t => t.ObjectiveValue).First()
            : completedTrials.OrderBy(t => t.ObjectiveValue).First();
    }

    /// <summary>
    /// Gets all trials performed during optimization.
    /// </summary>
    public List<HyperparameterTrial<T>> GetAllTrials()
    {
        return new List<HyperparameterTrial<T>>(_trials);
    }

    /// <summary>
    /// Gets trials that meet a certain criteria.
    /// </summary>
    public List<HyperparameterTrial<T>> GetTrials(Func<HyperparameterTrial<T>, bool> filter)
    {
        if (filter == null)
            throw new ArgumentNullException(nameof(filter));

        return _trials.Where(filter).ToList();
    }

    /// <summary>
    /// Suggests the next hyperparameter configuration to try.
    /// </summary>
    /// <remarks>
    /// For random search, this simply samples randomly from the search space.
    /// </remarks>
    public Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial)
    {
        if (_searchSpace == null)
            throw new InvalidOperationException("Search space not initialized. Call Optimize() first.");

        var parameters = new Dictionary<string, object>();

        foreach (var param in _searchSpace.Parameters)
        {
            parameters[param.Key] = param.Value.Sample(_random);
        }

        return parameters;
    }

    /// <summary>
    /// Reports the result of a trial.
    /// </summary>
    public void ReportTrial(HyperparameterTrial<T> trial, T objectiveValue)
    {
        if (trial == null)
            throw new ArgumentNullException(nameof(trial));

        trial.Complete(objectiveValue);
    }

    /// <summary>
    /// Determines if a trial should be pruned (stopped early).
    /// </summary>
    /// <remarks>
    /// Random search doesn't use pruning, so this always returns false.
    /// </remarks>
    public bool ShouldPrune(HyperparameterTrial<T> trial, int step, T intermediateValue)
    {
        // Random search doesn't prune trials
        return false;
    }
}
