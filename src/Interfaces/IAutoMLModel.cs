using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines the contract for AutoML models that automatically search for optimal model configurations.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type</typeparam>
    /// <typeparam name="TOutput">The output data type</typeparam>
    /// <remarks>
    /// AutoML (Automated Machine Learning) models automatically search through different model types,
    /// hyperparameters, and architectures to find the best configuration for a given dataset.
    /// This interface extends IFullModel to provide AutoML-specific functionality like search space configuration,
    /// trial management, and optimization settings.
    /// </remarks>
    public interface IAutoMLModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
    {
        /// <summary>
        /// Gets the current optimization status
        /// </summary>
        AutoMLStatus Status { get; }

        /// <summary>
        /// Gets the best model found so far
        /// </summary>
        IFullModel<T, TInput, TOutput>? BestModel { get; }

        /// <summary>
        /// Gets the best score achieved
        /// </summary>
        double BestScore { get; }

        /// <summary>
        /// Gets or sets the time limit for the AutoML search
        /// </summary>
        TimeSpan TimeLimit { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of trials to run
        /// </summary>
        int TrialLimit { get; set; }

        /// <summary>
        /// Searches for the best model configuration asynchronously
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        /// <param name="timeLimit">Time limit for the search</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The best model found</returns>
        Task<IFullModel<T, TInput, TOutput>> SearchAsync(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Sets the search space for hyperparameters
        /// </summary>
        /// <param name="searchSpace">Dictionary defining parameter ranges to search</param>
        void SetSearchSpace(Dictionary<string, ParameterRange> searchSpace);

        /// <summary>
        /// Configures the search space for hyperparameter optimization
        /// </summary>
        /// <param name="searchSpace">Dictionary defining parameter ranges to search</param>
        void ConfigureSearchSpace(Dictionary<string, ParameterRange> searchSpace);

        /// <summary>
        /// Sets the models to consider in the search
        /// </summary>
        /// <param name="modelTypes">List of model types to evaluate</param>
        void SetCandidateModels(List<ModelType> modelTypes);

        /// <summary>
        /// Sets which model types should be considered during the search
        /// </summary>
        /// <param name="modelTypes">List of model types to evaluate</param>
        void SetModelsToTry(List<ModelType> modelTypes);

        /// <summary>
        /// Sets the optimization metric
        /// </summary>
        /// <param name="metric">The metric to optimize</param>
        /// <param name="maximize">Whether to maximize (true) or minimize (false) the metric</param>
        void SetOptimizationMetric(MetricType metric, bool maximize = true);

        /// <summary>
        /// Gets the history of all trials
        /// </summary>
        /// <returns>List of trial results</returns>
        List<TrialResult> GetTrialHistory();

        /// <summary>
        /// Gets the results of all trials performed during search
        /// </summary>
        /// <returns>List of trial results with scores and parameters</returns>
        List<TrialResult> GetResults();

        /// <summary>
        /// Gets feature importance from the best model
        /// </summary>
        /// <returns>Dictionary mapping feature indices to importance scores</returns>
        Task<Dictionary<int, double>> GetFeatureImportanceAsync();

        /// <summary>
        /// Suggests the next hyperparameters to try
        /// </summary>
        /// <returns>Dictionary of suggested parameter values</returns>
        Task<Dictionary<string, object>> SuggestNextTrialAsync();

        /// <summary>
        /// Reports the result of a trial
        /// </summary>
        /// <param name="parameters">The parameters used in the trial</param>
        /// <param name="score">The score achieved</param>
        /// <param name="duration">The duration of the trial</param>
        Task ReportTrialResultAsync(Dictionary<string, object> parameters, double score, TimeSpan duration);

        /// <summary>
        /// Enables early stopping
        /// </summary>
        /// <param name="patience">Number of trials without improvement before stopping</param>
        /// <param name="minDelta">Minimum change to be considered an improvement</param>
        void EnableEarlyStopping(int patience, double minDelta = 0.001);

        /// <summary>
        /// Sets constraints for the search
        /// </summary>
        /// <param name="constraints">List of search constraints</param>
        void SetConstraints(List<SearchConstraint> constraints);

        /// <summary>
        /// Sets the time limit for the AutoML search process
        /// </summary>
        /// <param name="timeLimit">Maximum time to spend searching for optimal models</param>
        void SetTimeLimit(TimeSpan timeLimit);

        /// <summary>
        /// Sets the maximum number of trials to execute during search
        /// </summary>
        /// <param name="maxTrials">Maximum number of model configurations to try</param>
        void SetTrialLimit(int maxTrials);

        /// <summary>
        /// Enables Neural Architecture Search (NAS) for automatic network design
        /// </summary>
        /// <param name="enabled">Whether to enable NAS</param>
        void EnableNAS(bool enabled = true);

        /// <summary>
        /// Searches for the best model configuration (synchronous version)
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        /// <returns>Best model found</returns>
        IFullModel<T, TInput, TOutput> SearchBestModel(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets);

        /// <summary>
        /// Performs the AutoML search process (synchronous version)
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        void Search(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets);

        /// <summary>
        /// Runs the AutoML optimization process (alternative name for Search)
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        void Run(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets);
    }
}
