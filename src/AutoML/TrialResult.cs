using System;
using System.Collections.Generic;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents the result of a single trial during AutoML search
    /// </summary>
    public class TrialResult
    {
        /// <summary>
        /// Unique identifier for the trial
        /// </summary>
        public int TrialId { get; set; }

        /// <summary>
        /// The hyperparameters used in this trial
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// The score achieved by this trial
        /// </summary>
        public double Score { get; set; }

        /// <summary>
        /// The duration of the trial
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Timestamp when the trial was completed
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Additional metadata about the trial
        /// </summary>
        public Dictionary<string, object>? Metadata { get; set; }

        /// <summary>
        /// Whether the trial completed successfully
        /// </summary>
        public bool Success { get; set; } = true;

        /// <summary>
        /// Error message if the trial failed
        /// </summary>
        public string? ErrorMessage { get; set; }

        /// <summary>
        /// Creates a deep copy of the TrialResult
        /// </summary>
        public TrialResult Clone()
        {
            return new TrialResult
            {
                TrialId = TrialId,
                Parameters = new Dictionary<string, object>(Parameters),
                Score = Score,
                Duration = Duration,
                Timestamp = Timestamp,
                Metadata = Metadata != null ? new Dictionary<string, object>(Metadata) : null,
                Success = Success,
                ErrorMessage = ErrorMessage
            };
        }
    }
}
