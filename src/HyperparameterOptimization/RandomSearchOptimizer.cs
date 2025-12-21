using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.Helpers;

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
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public class RandomSearchOptimizer<T, TInput, TOutput> : HyperparameterOptimizerBase<T, TInput, TOutput>
{
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the RandomSearchOptimizer class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses a random seed.</param>
    public RandomSearchOptimizer(bool maximize = true, int? seed = null) : base(maximize)
    {
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration.
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

        lock (SyncLock)
        {
            for (int i = 0; i < nTrials; i++)
            {
                var trial = new HyperparameterTrial<T>(i);

                // Sample hyperparameters
                var parameters = SuggestNext(trial);

                // Evaluate with safe exception handling
                EvaluateTrialSafely(trial, objectiveFunction, parameters);

                Trials.Add(trial);
            }
        }

        var endTime = DateTime.UtcNow;

        return CreateOptimizationResult(searchSpace, startTime, endTime, nTrials);
    }

    /// <summary>
    /// Suggests the next hyperparameter configuration to try.
    /// </summary>
    /// <remarks>
    /// For random search, this simply samples randomly from the search space.
    /// </remarks>
    public override Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial)
    {
        if (SearchSpace == null)
            throw new InvalidOperationException("Search space not initialized. Call Optimize() first.");

        var parameters = new Dictionary<string, object>();

        foreach (var param in SearchSpace.Parameters)
        {
            parameters[param.Key] = param.Value.Sample(_random);
        }

        return parameters;
    }
}
