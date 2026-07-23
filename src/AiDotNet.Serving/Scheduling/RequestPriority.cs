namespace AiDotNet.Serving.Scheduling;

/// <summary>
/// Enumeration of request priority levels for scheduling.
/// </summary>
public enum RequestPriority
{
    /// <summary>
    /// Low priority request (e.g., batch processing, analytics)
    /// </summary>
    Low = 0,

    /// <summary>
    /// Normal priority request (default)
    /// </summary>
    Normal = 1,

    /// <summary>
    /// High priority request (e.g., user-facing real-time predictions)
    /// </summary>
    High = 2,

    /// <summary>
    /// Critical priority request (e.g., system health checks, emergency predictions)
    /// </summary>
    Critical = 3
}
