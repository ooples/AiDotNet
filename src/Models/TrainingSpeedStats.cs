namespace AiDotNet.Models;

/// <summary>
/// Contains statistics about training speed and progress.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This tracks how fast your model is training and estimates
/// how long until it's done.
/// </remarks>
public class TrainingSpeedStats
{
    /// <summary>
    /// Gets or sets the average iterations per second.
    /// </summary>
    public double IterationsPerSecond { get; set; }

    /// <summary>
    /// Gets or sets the average time per iteration in seconds.
    /// </summary>
    public double SecondsPerIteration { get; set; }

    /// <summary>
    /// Gets or sets the estimated time remaining.
    /// </summary>
    public TimeSpan EstimatedTimeRemaining { get; set; }

    /// <summary>
    /// Gets or sets the total elapsed time.
    /// </summary>
    public TimeSpan ElapsedTime { get; set; }

    /// <summary>
    /// Gets or sets the current progress percentage (0-100).
    /// </summary>
    public double ProgressPercentage { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations completed.
    /// </summary>
    public int IterationsCompleted { get; set; }

    /// <summary>
    /// Gets or sets the total number of iterations planned.
    /// </summary>
    public int TotalIterations { get; set; }
}
