namespace AiDotNet.Enums;

/// <summary>
/// Represents the current status of an AutoML search process.
/// </summary>
public enum AutoMLStatus
{
    /// <summary>
    /// The AutoML search has not started yet.
    /// </summary>
    NotStarted,

    /// <summary>
    /// The AutoML search is currently running.
    /// </summary>
    Running,

    /// <summary>
    /// The AutoML search has completed successfully.
    /// </summary>
    Completed,

    /// <summary>
    /// The AutoML search was cancelled.
    /// </summary>
    Cancelled,

    /// <summary>
    /// The AutoML search failed with an error.
    /// </summary>
    Failed
}
