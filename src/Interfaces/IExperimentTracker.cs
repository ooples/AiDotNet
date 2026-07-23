namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for experiment tracking systems that log machine learning experiments.
/// </summary>
/// <remarks>
/// An experiment tracker records information about machine learning experiments including parameters,
/// metrics, and artifacts to enable reproducibility and comparison of different training runs.
///
/// <b>For Beginners:</b> Think of an experiment tracker as a lab notebook for machine learning.
/// Just like a scientist records their experimental conditions and results, an experiment tracker
/// logs all the details of your machine learning model training - what settings you used, how well
/// it performed, and what models you created.
///
/// Key capabilities include:
/// - Creating and managing experiments (groups of related training runs)
/// - Logging hyperparameters (settings used for training)
/// - Recording metrics (performance measurements over time)
/// - Storing artifacts (models, plots, data files)
/// - Comparing different training runs
/// - Reproducing previous experiments
///
/// Why experiment tracking matters:
/// - Helps you keep track of what you've tried
/// - Makes it easy to reproduce good results
/// - Enables comparison between different approaches
/// - Provides audit trail for model development
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IExperimentTracker<T>
{
    /// <summary>
    /// Creates a new experiment to organize related training runs.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> An experiment is a collection of related training runs.
    /// For example, you might create an experiment called "House Price Prediction"
    /// and then have multiple runs within it trying different algorithms or settings.
    /// </remarks>
    /// <param name="name">The name of the experiment.</param>
    /// <param name="description">Optional description of the experiment's purpose.</param>
    /// <param name="tags">Optional tags to categorize the experiment.</param>
    /// <returns>The unique identifier for the created experiment.</returns>
    string CreateExperiment(string name, string? description = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Starts a new training run within an experiment.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A run represents a single training session with specific settings.
    /// Each time you train a model with different parameters, you start a new run.
    /// </remarks>
    /// <param name="experimentId">The ID of the experiment this run belongs to.</param>
    /// <param name="runName">Optional name for this specific run.</param>
    /// <param name="tags">Optional tags to categorize the run.</param>
    /// <returns>An IExperimentRun object for logging metrics and parameters.</returns>
    IExperimentRun<T> StartRun(string experimentId, string? runName = null, Dictionary<string, string>? tags = null);

    /// <summary>
    /// Gets an existing experiment by its ID.
    /// </summary>
    /// <param name="experimentId">The unique identifier of the experiment.</param>
    /// <returns>An IExperiment object containing experiment details.</returns>
    IExperiment GetExperiment(string experimentId);

    /// <summary>
    /// Gets an existing run by its ID.
    /// </summary>
    /// <param name="runId">The unique identifier of the run.</param>
    /// <returns>An IExperimentRun object containing run details.</returns>
    IExperimentRun<T> GetRun(string runId);

    /// <summary>
    /// Lists all experiments, optionally filtered by criteria.
    /// </summary>
    /// <param name="filter">Optional filter expression.</param>
    /// <returns>A list of experiments matching the criteria.</returns>
    IEnumerable<IExperiment> ListExperiments(string? filter = null);

    /// <summary>
    /// Lists all runs in an experiment, optionally filtered by criteria.
    /// </summary>
    /// <param name="experimentId">The experiment ID to list runs from.</param>
    /// <param name="filter">Optional filter expression.</param>
    /// <returns>A list of runs matching the criteria.</returns>
    IEnumerable<IExperimentRun<T>> ListRuns(string experimentId, string? filter = null);

    /// <summary>
    /// Deletes an experiment and all its associated runs.
    /// </summary>
    /// <param name="experimentId">The ID of the experiment to delete.</param>
    void DeleteExperiment(string experimentId);

    /// <summary>
    /// Deletes a specific run.
    /// </summary>
    /// <param name="runId">The ID of the run to delete.</param>
    void DeleteRun(string runId);

    /// <summary>
    /// Searches for runs across all experiments based on criteria.
    /// </summary>
    /// <param name="filter">Filter expression for searching runs.</param>
    /// <param name="maxResults">Maximum number of results to return.</param>
    /// <returns>A list of runs matching the search criteria.</returns>
    IEnumerable<IExperimentRun<T>> SearchRuns(string filter, int maxResults = 100);
}
