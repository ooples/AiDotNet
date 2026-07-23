namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for hyperparameter optimization algorithms.
/// </summary>
/// <remarks>
/// A hyperparameter optimizer automatically searches for the best hyperparameters for a machine learning model
/// by trying different combinations and evaluating their performance.
///
/// <b>For Beginners:</b> Think of hyperparameters as the "settings" for your machine learning algorithm
/// (like learning rate, number of layers, etc.). A hyperparameter optimizer is like an automatic tuner that
/// tries different settings to find the combination that works best for your data.
///
/// Common optimization strategies include:
/// - Grid Search: Tries every possible combination in a predefined grid
/// - Random Search: Randomly samples combinations
/// - Bayesian Optimization: Uses past results to intelligently choose what to try next
/// - Hyperband: Efficiently allocates resources to promising configurations
///
/// Why hyperparameter optimization matters:
/// - Manual tuning is time-consuming and error-prone
/// - Good hyperparameters can dramatically improve model performance
/// - Systematic search ensures you don't miss good configurations
/// - Enables reproducible model selection
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IHyperparameterOptimizer<T, TInput, TOutput>
{
    /// <summary>
    /// Searches for the best hyperparameter configuration.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main method that performs the hyperparameter search.
    /// It tries different combinations of hyperparameters and returns the best one found.
    /// </remarks>
    /// <param name="objectiveFunction">The function to optimize (typically model performance).</param>
    /// <param name="searchSpace">The space of possible hyperparameter values to search.</param>
    /// <param name="nTrials">Number of trials to run.</param>
    /// <returns>The best hyperparameter configuration found.</returns>
    HyperparameterOptimizationResult<T> Optimize(
        Func<Dictionary<string, object>, T> objectiveFunction,
        HyperparameterSearchSpace searchSpace,
        int nTrials);

    /// <summary>
    /// Searches for the best hyperparameters for a specific model.
    /// </summary>
    /// <param name="model">The model to optimize hyperparameters for.</param>
    /// <param name="trainingData">The training data to use for evaluation.</param>
    /// <param name="validationData">The validation data to use for evaluation.</param>
    /// <param name="searchSpace">The space of possible hyperparameter values.</param>
    /// <param name="nTrials">Number of trials to run.</param>
    /// <returns>The optimized model with best hyperparameters.</returns>
    HyperparameterOptimizationResult<T> OptimizeModel<TMetadata>(
        IModel<TInput, TOutput, TMetadata> model,
        (TInput X, TOutput Y) trainingData,
        (TInput X, TOutput Y) validationData,
        HyperparameterSearchSpace searchSpace,
        int nTrials) where TMetadata : class;

    /// <summary>
    /// Gets the best trial from the optimization.
    /// </summary>
    /// <returns>The trial with the best objective value.</returns>
    HyperparameterTrial<T> GetBestTrial();

    /// <summary>
    /// Gets all trials performed during optimization.
    /// </summary>
    /// <returns>List of all trials.</returns>
    List<HyperparameterTrial<T>> GetAllTrials();

    /// <summary>
    /// Gets trials that meet a certain criteria.
    /// </summary>
    /// <param name="filter">Filter function to select trials.</param>
    /// <returns>Filtered list of trials.</returns>
    List<HyperparameterTrial<T>> GetTrials(Func<HyperparameterTrial<T>, bool> filter);

    /// <summary>
    /// Suggests the next hyperparameter configuration to try based on past trials.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Advanced optimizers (like Bayesian optimization) learn from
    /// previous trials to intelligently choose what to try next. This method provides that suggestion.
    /// </remarks>
    /// <param name="trial">The trial to populate with suggestions.</param>
    /// <returns>Dictionary of suggested hyperparameter values.</returns>
    Dictionary<string, object> SuggestNext(HyperparameterTrial<T> trial);

    /// <summary>
    /// Reports the result of a trial.
    /// </summary>
    /// <param name="trial">The trial to report results for.</param>
    /// <param name="objectiveValue">The objective value achieved.</param>
    void ReportTrial(HyperparameterTrial<T> trial, T objectiveValue);

    /// <summary>
    /// Determines if a trial should be pruned (stopped early) to save resources.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Pruning is stopping a trial early if it's clearly not performing well,
    /// which saves time and computational resources. It's like stopping a cake from baking if you
    /// can already tell it's burned.
    /// </remarks>
    /// <param name="trial">The trial to check.</param>
    /// <param name="step">The current training step.</param>
    /// <param name="intermediateValue">The current performance value.</param>
    /// <returns>True if the trial should be stopped early.</returns>
    bool ShouldPrune(HyperparameterTrial<T> trial, int step, T intermediateValue);
}
