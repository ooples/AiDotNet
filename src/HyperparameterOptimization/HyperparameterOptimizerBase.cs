using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.HyperparameterOptimization;

/// <summary>
/// Base class for hyperparameter optimization algorithms.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This abstract base class provides common functionality for hyperparameter
/// optimization. It handles trial management, result tracking, and provides helper methods for
/// finding best trials while leaving the specific optimization strategy to derived classes.
///
/// Key features:
/// - Thread-safe trial management
/// - Consistent result aggregation
/// - Helper methods for finding best trials
/// - Support for both maximization and minimization objectives
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for models.</typeparam>
/// <typeparam name="TOutput">The output data type for models.</typeparam>
public abstract class HyperparameterOptimizerBase<T, TInput, TOutput> : IHyperparameterOptimizer<T, TInput, TOutput>
{
    /// <summary>
    /// Collection of all trials performed during optimization.
    /// </summary>
    protected readonly List<HyperparameterTrial<T>> Trials;

    /// <summary>
    /// Whether to maximize (true) or minimize (false) the objective function.
    /// </summary>
    protected readonly bool Maximize;

    /// <summary>
    /// The search space being optimized.
    /// </summary>
    protected HyperparameterSearchSpace? SearchSpace;

    /// <summary>
    /// Lock object for thread-safe operations.
    /// </summary>
    protected readonly object SyncLock = new();

    /// <summary>
    /// Initializes a new instance of the HyperparameterOptimizerBase class.
    /// </summary>
    /// <param name="maximize">Whether to maximize the objective (true) or minimize it (false).</param>
    protected HyperparameterOptimizerBase(bool maximize = true)
    {
        Trials = new List<HyperparameterTrial<T>>();
        Maximize = maximize;
    }

    /// <summary>
    /// Searches for the best hyperparameter configuration.
    /// </summary>
    public abstract HyperparameterOptimizationResult<T> Optimize(
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        int nTrials);

    /// <summary>
    /// Searches for the best hyperparameters for a specific model.
    /// </summary>
    public virtual HyperparameterOptimizationResult<T> OptimizeModel<TMetadata>(
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
    public virtual HyperparameterTrial<T> GetBestTrial()
    {
        lock (SyncLock)
        {
            var completedTrials = Trials.Where(t => t.Status == TrialStatus.Complete).ToList();
            if (completedTrials.Count == 0)
                throw new InvalidOperationException("No completed trials available.");

            return FindBestTrial(completedTrials);
        }
    }

    /// <summary>
    /// Gets all trials performed during optimization.
    /// </summary>
    public virtual List<HyperparameterTrial<T>> GetAllTrials()
    {
        lock (SyncLock)
        {
            return new List<HyperparameterTrial<T>>(Trials);
        }
    }

    /// <summary>
    /// Gets trials that meet a certain criteria.
    /// </summary>
    public virtual List<HyperparameterTrial<T>> GetTrials(Func<HyperparameterTrial<T>, bool> filter)
    {
        if (filter == null)
            throw new ArgumentNullException(nameof(filter));

        lock (SyncLock)
        {
            return Trials.Where(filter).ToList();
        }
    }

    /// <summary>
    /// Suggests the next hyperparameter configuration to try.
    /// </summary>
    public abstract Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial);

    /// <summary>
    /// Reports the result of a trial.
    /// </summary>
    public virtual void ReportTrial(HyperparameterTrial<T> trial, T objectiveValue)
    {
        if (trial == null)
            throw new ArgumentNullException(nameof(trial));

        trial.Complete(objectiveValue);
    }

    /// <summary>
    /// Determines if a trial should be pruned (stopped early).
    /// </summary>
    /// <remarks>
    /// Default implementation returns false. Override in derived classes to implement pruning.
    /// </remarks>
    public virtual bool ShouldPrune(HyperparameterTrial<T> trial, int step, T intermediateValue)
    {
        return false;
    }

    #region Protected Helper Methods

    /// <summary>
    /// Finds the best trial from a list of completed trials.
    /// </summary>
    /// <param name="completedTrials">The list of completed trials to search.</param>
    /// <returns>The trial with the best objective value.</returns>
    protected virtual HyperparameterTrial<T> FindBestTrial(List<HyperparameterTrial<T>> completedTrials)
    {
        return Maximize
            ? completedTrials.OrderByDescending(t => t.ObjectiveValue).First()
            : completedTrials.OrderBy(t => t.ObjectiveValue).First();
    }

    /// <summary>
    /// Creates an optimization result from the current trials.
    /// </summary>
    /// <param name="searchSpace">The search space used.</param>
    /// <param name="startTime">The start time of optimization.</param>
    /// <param name="endTime">The end time of optimization.</param>
    /// <param name="totalTrials">The total number of trials planned.</param>
    /// <returns>The optimization result.</returns>
    protected virtual HyperparameterOptimizationResult<T> CreateOptimizationResult(
        HyperparameterSearchSpace searchSpace,
        DateTime startTime,
        DateTime endTime,
        int totalTrials)
    {
        var completedTrials = Trials.Where(t => t.Status == TrialStatus.Complete).ToList();
        if (completedTrials.Count == 0)
            throw new InvalidOperationException("No trials completed successfully.");

        var bestTrial = FindBestTrial(completedTrials);

        return new HyperparameterOptimizationResult<T>
        {
            BestTrial = bestTrial,
            AllTrials = new List<HyperparameterTrial<T>>(Trials),
            BestParameters = bestTrial.Parameters,
            BestObjectiveValue = bestTrial.ObjectiveValue!,
            SearchSpace = searchSpace,
            TotalTrials = totalTrials,
            CompletedTrials = Trials.Count(t => t.Status == TrialStatus.Complete),
            PrunedTrials = Trials.Count(t => t.Status == TrialStatus.Pruned),
            FailedTrials = Trials.Count(t => t.Status == TrialStatus.Failed),
            TotalTime = endTime - startTime,
            StartTime = startTime,
            EndTime = endTime
        };
    }

    /// <summary>
    /// Evaluates a trial with the objective function and handles exceptions.
    /// </summary>
    /// <param name="trial">The trial to evaluate.</param>
    /// <param name="objectiveFunction">The objective function.</param>
    /// <param name="parameters">The hyperparameters to evaluate.</param>
    protected virtual void EvaluateTrialSafely(
        HyperparameterTrial<T> trial,
        Func<Dictionary<string, object>, T> objectiveFunction,
        Dictionary<string, object> parameters)
    {
        trial.Parameters = parameters;

        try
        {
            var objectiveValue = objectiveFunction(parameters);
            trial.Complete(objectiveValue);
        }
        catch (OutOfMemoryException)
        {
            throw;
        }
        catch (StackOverflowException)
        {
            throw;
        }
        catch (ArgumentException ex)
        {
            trial.Fail();
            trial.UserAttributes["error"] = $"ArgumentException: {ex.Message}";
        }
        catch (InvalidOperationException ex)
        {
            trial.Fail();
            trial.UserAttributes["error"] = $"InvalidOperationException: {ex.Message}";
        }
        catch (TimeoutException ex)
        {
            trial.Fail();
            trial.UserAttributes["error"] = $"TimeoutException: {ex.Message}";
        }
        catch (AggregateException ex)
        {
            trial.Fail();
            trial.UserAttributes["error"] = $"AggregateException: {ex.InnerException?.Message ?? ex.Message}";
        }
        catch (Exception ex)
        {
            // Catch-all for any other exceptions from user-provided objective functions
            // This ensures the optimization loop continues even if a trial fails unexpectedly
            trial.Fail();
            trial.UserAttributes["error"] = $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    /// <summary>
    /// Validates the optimization inputs.
    /// </summary>
    /// <param name="objectiveFunction">The objective function to validate.</param>
    /// <param name="searchSpace">The search space to validate.</param>
    /// <param name="nTrials">The number of trials to validate.</param>
    protected virtual void ValidateOptimizationInputs(
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
    }

    #endregion
}
