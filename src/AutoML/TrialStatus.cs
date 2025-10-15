namespace AiDotNet.AutoML
{
    /// <summary>
    /// Represents the status of an AutoML trial
    /// </summary>
    public enum TrialStatus
    {
        /// <summary>
        /// The trial is pending execution
        /// </summary>
        Pending,

        /// <summary>
        /// The trial is currently running
        /// </summary>
        Running,

        /// <summary>
        /// The trial completed successfully
        /// </summary>
        Completed,

        /// <summary>
        /// The trial failed with an error
        /// </summary>
        Failed,

        /// <summary>
        /// The trial was cancelled
        /// </summary>
        Cancelled,

        /// <summary>
        /// The trial was skipped due to constraints
        /// </summary>
        Skipped
    }
}