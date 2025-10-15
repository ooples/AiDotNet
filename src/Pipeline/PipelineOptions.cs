namespace AiDotNet.Pipeline
{
    /// <summary>
    /// General options for configuring pipeline execution behavior.
    /// </summary>
    public class PipelineOptions
    {
        /// <summary>
        /// Gets or sets whether to enable parallel execution of independent pipeline steps.
        /// </summary>
        public bool EnableParallelExecution { get; set; } = false;

        /// <summary>
        /// Gets or sets the maximum degree of parallelism when parallel execution is enabled.
        /// </summary>
        public int MaxDegreeOfParallelism { get; set; } = 4;

        /// <summary>
        /// Gets or sets whether to cache intermediate results between pipeline steps.
        /// </summary>
        public bool CacheIntermediateResults { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to automatically save checkpoints during pipeline execution.
        /// </summary>
        public bool EnableAutoCheckpointing { get; set; } = true;

        /// <summary>
        /// Gets or sets the checkpoint interval (number of steps between checkpoints).
        /// </summary>
        public int CheckpointInterval { get; set; } = 5;

        /// <summary>
        /// Gets or sets whether to enable detailed logging of pipeline operations.
        /// </summary>
        public bool EnableVerboseLogging { get; set; } = false;

        /// <summary>
        /// Gets or sets whether to continue execution on non-critical errors.
        /// </summary>
        public bool ContinueOnError { get; set; } = false;

        /// <summary>
        /// Gets or sets the timeout in seconds for individual pipeline steps (0 = no timeout).
        /// </summary>
        public int StepTimeoutSeconds { get; set; } = 0;

        /// <summary>
        /// Gets or sets whether to validate data integrity between pipeline steps.
        /// </summary>
        public bool ValidateDataIntegrity { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to enable GPU acceleration when available.
        /// </summary>
        public bool EnableGPUAcceleration { get; set; } = false;
    }
}
