using AiDotNet.Enums;
using System;
using System.Collections.Generic;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents the result of a single AutoML trial
    /// </summary>
    public class TrialResult
    {
        /// <summary>
        /// Gets or sets the unique identifier for this trial
        /// </summary>
        public int TrialId { get; set; }

        /// <summary>
        /// Gets or sets the hyperparameters used in this trial
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new();

        /// <summary>
        /// Gets or sets the hyperparameters used in this trial (alias for Parameters for backward compatibility)
        /// </summary>
        public Dictionary<string, object> Hyperparameters
        {
            get => Parameters;
            set => Parameters = value;
        }

        /// <summary>
        /// Gets or sets the score achieved in this trial
        /// </summary>
        public double Score { get; set; }

        /// <summary>
        /// Gets or sets the duration of this trial
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Gets or sets the type of model used in this trial
        /// </summary>
        public ModelType ModelType { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when this trial was completed
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Gets or sets the error message if the trial failed
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// Gets or sets the status of this trial
        /// </summary>
        public TrialStatus Status { get; set; } = TrialStatus.Pending;

        /// <summary>
        /// Gets or sets whether this trial was successful
        /// </summary>
        public bool IsSuccessful => Status == TrialStatus.Completed;

        /// <summary>
        /// Gets or sets additional metrics collected during the trial
        /// </summary>
        public Dictionary<string, double> AdditionalMetrics { get; set; } = new();

        /// <summary>
        /// Creates a copy of this trial result
        /// </summary>
        /// <returns>A deep copy of the trial result</returns>
        public TrialResult Clone()
        {
            return new TrialResult
            {
                TrialId = TrialId,
                Parameters = new Dictionary<string, object>(Parameters),
                Score = Score,
                Duration = Duration,
                ModelType = ModelType,
                Timestamp = Timestamp,
                Status = Status,
                ErrorMessage = ErrorMessage,
                AdditionalMetrics = new Dictionary<string, double>(AdditionalMetrics)
            };
        }

        /// <summary>
        /// Returns a string representation of this trial result
        /// </summary>
        public override string ToString()
        {
            return $"Trial {TrialId}: Score={Score:F4}, Model={ModelType}, Duration={Duration.TotalSeconds:F2}s, Success={IsSuccessful}";
        }
    }
}