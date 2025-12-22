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
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class GridSearchOptimizer<T, TInput, TOutput> : HyperparameterOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the GridSearchOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    public GridSearchOptimizer(bool maximize = true) : base(maximize)
    {
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration using grid search.
    /// </summary>
    public override HyperparameterOptimizationResult<T> Optimize(
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        int nTrials)
    {
        ValidateOptimizationInputs(objectiveFunction, searchSpace, nTrials);

        SearchSpace = searchSpace;
        Trials.Clear(); // Clear trials from previous Optimize calls
        var startTime = DateTime.UtcNow;

        // Generate all combinations
        var allCombinations = GenerateAllCombinations(searchSpace);

        // Limit to nTrials if specified
        var combinationsToTry = nTrials > 0
            ? allCombinations.Take(nTrials).ToList()
            : allCombinations;

        lock (SyncLock)
        {
            int trialNumber = 0;
            foreach (var parameters in combinationsToTry)
            {
                var trial = new HyperparameterTrial<T>(trialNumber++);

                // Evaluate with safe exception handling
                EvaluateTrialSafely(trial, objectiveFunction, parameters);

                Trials.Add(trial);
            }
        }

        var endTime = DateTime.UtcNow;

        return CreateOptimizationResult(searchSpace, startTime, endTime, Trials.Count);
    }

    /// <summary>
    /// Grid search doesn't use suggestions - it pre-generates all combinations.
    /// </summary>
    public override Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial)
    {
        throw new NotSupportedException("Grid search doesn't use suggestions. All combinations are pre-generated.");
    }

    #region Private Helper Methods

    private List<Dictionary<string, object>> GenerateAllCombinations(HyperparameterSearchSpace searchSpace)
    {
        var parameterNames = searchSpace.Parameters.Keys.ToList();
        var parameterValues = new List<List<object>>();

        // For continuous distributions, sample a discrete set of values
        parameterValues.AddRange(parameterNames
            .Select(paramName => GetDiscreteValues(searchSpace.Parameters[paramName])));

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
        // Validate distribution parameters to prevent infinite loops and silent failures
        if (distribution.Step <= 0)
        {
            throw new ArgumentException(
                $"Invalid distribution: Step must be > 0, but got {distribution.Step}.",
                nameof(distribution));
        }

        if (distribution.Min > distribution.Max)
        {
            throw new ArgumentException(
                $"Invalid distribution: Min ({distribution.Min}) must be <= Max ({distribution.Max}).",
                nameof(distribution));
        }

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

        if (numSamples <= 0)
            throw new ArgumentException("numSamples must be positive.", nameof(numSamples));

        // Handle single sample case to avoid division by zero
        if (numSamples == 1)
        {
            if (distribution.LogScale)
            {
                // Return geometric mean for log scale
                values.Add(Math.Exp((Math.Log(distribution.Min) + Math.Log(distribution.Max)) / 2));
            }
            else
            {
                // Return arithmetic midpoint
                values.Add((distribution.Min + distribution.Max) / 2);
            }
            return values;
        }

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
