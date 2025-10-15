using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Pipeline
{
    /// <summary>
    /// Represents the result of a pipeline orchestrator execution
    /// </summary>
    /// <typeparam name="T">The numeric type for computations</typeparam>
    public class PipelineResult<T>
    {
        /// <summary>
        /// Gets or sets the unique identifier for this result
        /// </summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the orchestrator ID
        /// </summary>
        public string OrchestratorId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the end time
        /// </summary>
        public DateTime EndTime { get; set; }

        /// <summary>
        /// Gets or sets whether the execution was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the error message if failed
        /// </summary>
        public string? Error { get; set; }

        /// <summary>
        /// Gets or sets the main pipeline output
        /// </summary>
        public Matrix<T>? MainPipelineOutput { get; set; }

        /// <summary>
        /// Gets or sets the branch outputs
        /// </summary>
        public Dictionary<string, Matrix<T>>? BranchOutputs { get; set; }

        /// <summary>
        /// Gets or sets the final merged output
        /// </summary>
        public Matrix<T>? FinalOutput { get; set; }

        /// <summary>
        /// Gets the duration of the execution
        /// </summary>
        public TimeSpan Duration => EndTime - StartTime;

        /// <summary>
        /// Gets or sets additional metadata
        /// </summary>
        public Dictionary<string, object> Metadata { get; } = new Dictionary<string, object>();
    }
}