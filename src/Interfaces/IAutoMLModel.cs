using AiDotNet.AutoML;
using AiDotNet.Enums;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for AutoML capabilities that can automatically find the best model for any type of data
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>)</typeparam>
    /// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>)</typeparam>
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
        /// Searches for the best model configuration
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        /// <param name="timeLimit">Time limit for search</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Best model found</returns>
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
        /// <param name="searchSpace">Dictionary defining parameter ranges</param>
        void SetSearchSpace(Dictionary<string, ParameterRange> searchSpace);

        /// <summary>
        /// Sets the models to consider in the search
        /// </summary>
        /// <param name="modelTypes">List of model types to try</param>
        void SetCandidateModels(List<ModelType> modelTypes);

        /// <summary>
        /// Sets the optimization metric
        /// </summary>
        /// <param name="metric">Metric to optimize</param>
        /// <param name="maximize">Whether to maximize (true) or minimize (false)</param>
        void SetOptimizationMetric(MetricType metric, bool maximize = true);

        /// <summary>
        /// Gets the history of all trials
        /// </summary>
        /// <returns>List of trial results</returns>
        List<TrialResult> GetTrialHistory();

        /// <summary>
        /// Gets feature importance from the best model
        /// </summary>
        /// <returns>Feature importance scores</returns>
        Task<Dictionary<int, double>> GetFeatureImportanceAsync();

        /// <summary>
        /// Suggests the next hyperparameters to try
        /// </summary>
        /// <returns>Suggested hyperparameters</returns>
        Task<Dictionary<string, object>> SuggestNextTrialAsync();

        /// <summary>
        /// Reports the result of a trial
        /// </summary>
        /// <param name="parameters">Parameters used</param>
        /// <param name="score">Score achieved</param>
        /// <param name="duration">Time taken</param>
        Task ReportTrialResultAsync(Dictionary<string, object> parameters, double score, TimeSpan duration);

        /// <summary>
        /// Enables early stopping
        /// </summary>
        /// <param name="patience">Number of trials without improvement before stopping</param>
        /// <param name="minDelta">Minimum improvement to reset patience counter</param>
        void EnableEarlyStopping(int patience, double minDelta = 0.001);

        /// <summary>
        /// Sets constraints for the search
        /// </summary>
        /// <param name="constraints">List of constraints</param>
        void SetConstraints(List<SearchConstraint> constraints);
    }

    /// <summary>
    /// Represents a parameter range for hyperparameter search
    /// </summary>
    public class ParameterRange
    {
        public object MinValue { get; set; }
        public object MaxValue { get; set; }
        public ParameterType Type { get; set; }
        public object[]? CategoricalValues { get; set; }
        public bool LogScale { get; set; }
    }

    /// <summary>
    /// Parameter types for AutoML search
    /// </summary>
    public enum ParameterType
    {
        Continuous,
        Integer,
        Categorical,
        Boolean
    }

    /// <summary>
    /// AutoML optimization status
    /// </summary>
    public enum AutoMLStatus
    {
        NotStarted,
        Running,
        Paused,
        Completed,
        Failed,
        Cancelled
    }


    /// <summary>
    /// Search constraint for AutoML
    /// </summary>
    public class SearchConstraint
    {
        public string Name { get; set; } = string.Empty;
        public ConstraintType Type { get; set; }
        public object Value { get; set; } = new object();
    }

    /// <summary>
    /// Types of constraints for AutoML search
    /// </summary>
    public enum ConstraintType
    {
        MaxModelSize,
        MaxInferenceTime,
        MinAccuracy,
        MaxMemoryUsage,
        RequireInterpretability
    }
}