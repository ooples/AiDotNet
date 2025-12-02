using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Implements grid search hyperparameter optimization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Grid search systematically tries every possible combination
/// of hyperparameters from a predefined grid.
///
/// How it works:
/// 1. Define a grid of values for each hyperparameter
/// 2. Generate all possible combinations
/// 3. Try each combination in sequence
/// 4. Return the best configuration found
///
/// Advantages:
/// - Guaranteed to find the best combination in the grid
/// - Systematic and reproducible
///
/// Disadvantages:
/// - Can be very slow with many hyperparameters (combinatorial explosion)
/// - Wastes time on unpromising regions
/// - Only searches discrete values, not continuous ranges
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class GridSearchOptimizer<T, TInput, TOutput> : IHyperparameterOptimizer<T, TInput, TOutput>
{
    private readonly List<HyperparameterTrial<T>> _trials;
    private readonly bool _maximize;
    private HyperparameterSearchSpace? _searchSpace;

    /// <summary>
    /// Initializes a new instance of the GridSearchOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    public GridSearchOptimizer(bool maximize = true)
    {
        _trials = new List<HyperparameterTrial<T>>();
        _maximize = maximize;
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration using grid search.
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

        _searchSpace = searchSpace;
        var startTime = DateTime.UtcNow;

        // Generate all combinations
        var allCombinations = GenerateAllCombinations(searchSpace);

        // Limit to nTrials if specified
        var combinationsToTry = nTrials > 0
            ? allCombinations.Take(nTrials).ToList()
            : allCombinations;

        int trialNumber = 0;
        foreach (var parameters in combinationsToTry)
        {
            var trial = new HyperparameterTrial<T>(trialNumber++);
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
            TotalTrials = _trials.Count,
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
    /// Grid search doesn't use suggestions - it pre-generates all combinations.
    /// </summary>
    public Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial)
    {
        throw new NotSupportedException("Grid search doesn't use suggestions. All combinations are pre-generated.");
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
    /// Grid search doesn't use pruning.
    /// </summary>
    public bool ShouldPrune(HyperparameterTrial<T> trial, int step, T intermediateValue)
    {
        return false;
    }

    #region Private Helper Methods

    private List<Dictionary<string, object>> GenerateAllCombinations(HyperparameterSearchSpace searchSpace)
    {
        var parameterNames = searchSpace.Parameters.Keys.ToList();
        var parameterValues = new List<List<object>>();

        // For continuous distributions, sample a discrete set of values
        foreach (var paramName in parameterNames)
        {
            var distribution = searchSpace.Parameters[paramName];
            parameterValues.Add(GetDiscreteValues(distribution));
        }

        // Generate all combinations
        return GenerateCombinationsRecursive(parameterNames, parameterValues, 0, new Dictionary<string, object>());
    }

    private List<object> GetDiscreteValues(ParameterDistribution distribution)
    {
        return distribution switch
        {
            CategoricalDistribution cat => cat.Choices,
            IntegerDistribution intDist => GenerateIntegerRange(intDist),
            ContinuousDistribution contDist => GenerateContinuousRange(contDist),
            _ => throw new NotSupportedException($"Distribution type {distribution.GetType().Name} is not supported.")
        };
    }

    private List<object> GenerateIntegerRange(IntegerDistribution distribution)
    {
        var values = new List<object>();
        for (int i = distribution.Min; i <= distribution.Max; i += distribution.Step)
        {
            values.Add(i);
        }
        return values;
    }

    private List<object> GenerateContinuousRange(ContinuousDistribution distribution, int numSamples = 10)
    {
        var values = new List<object>();

        if (distribution.LogScale)
        {
            var logMin = Math.Log(distribution.Min);
            var logMax = Math.Log(distribution.Max);
            for (int i = 0; i < numSamples; i++)
            {
                var logValue = logMin + (logMax - logMin) * i / (numSamples - 1);
                values.Add(Math.Exp(logValue));
            }
        }
        else
        {
            for (int i = 0; i < numSamples; i++)
            {
                var value = distribution.Min + (distribution.Max - distribution.Min) * i / (numSamples - 1);
                values.Add(value);
            }
        }

        return values;
    }

    private List<Dictionary<string, object>> GenerateCombinationsRecursive(
        List<string> parameterNames,
        List<List<object>> parameterValues,
        int index,
        Dictionary<string, object> currentCombination)
    {
        if (index == parameterNames.Count)
        {
            return new List<Dictionary<string, object>> { new Dictionary<string, object>(currentCombination) };
        }

        var combinations = new List<Dictionary<string, object>>();
        var paramName = parameterNames[index];
        var values = parameterValues[index];

        foreach (var value in values)
        {
            currentCombination[paramName] = value;
            combinations.AddRange(GenerateCombinationsRecursive(parameterNames, parameterValues, index + 1, currentCombination));
            currentCombination.Remove(paramName);
        }

        return combinations;
    }

    #endregion
}
