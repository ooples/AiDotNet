namespace AiDotNet.Models.Options;

/// <summary>
/// Global early stopping configuration that spans multiple training stages.
/// </summary>
public class GlobalEarlyStoppingConfig : EarlyStoppingConfig
{
    /// <summary>
    /// Gets or sets the maximum total training time across all stages.
    /// </summary>
    public TimeSpan? MaxTotalTime { get; set; }

    /// <summary>
    /// Gets or sets whether to stop the entire pipeline if any stage fails.
    /// </summary>
    public bool StopOnStageFailure { get; set; } = true;

    /// <summary>
    /// Gets or sets the target metric value to achieve (stops when reached).
    /// </summary>
    public double? TargetMetricValue { get; set; }
}
