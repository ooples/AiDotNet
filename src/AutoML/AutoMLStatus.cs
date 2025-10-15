namespace AiDotNet.AutoML
{
    /// <summary>
    /// AutoML optimization status
    /// </summary>
    public enum AutoMLStatus
    {
        /// <summary>
        /// The AutoML process has not been started yet
        /// </summary>
        NotStarted,
        
        /// <summary>
        /// The AutoML process is currently running
        /// </summary>
        Running,
        
        /// <summary>
        /// The AutoML process has been paused
        /// </summary>
        Paused,
        
        /// <summary>
        /// The AutoML process has completed successfully
        /// </summary>
        Completed,
        
        /// <summary>
        /// The AutoML process has failed with an error
        /// </summary>
        Failed,
        
        /// <summary>
        /// The AutoML process was cancelled by the user
        /// </summary>
        Cancelled
    }
}