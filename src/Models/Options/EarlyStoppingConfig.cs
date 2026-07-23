namespace AiDotNet.Models.Options;

/// <summary>
/// Early stopping configuration for a training stage.
/// </summary>
public class EarlyStoppingConfig
{
    /// <summary>
    /// Gets or sets whether early stopping is enabled.
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the metric to monitor for early stopping.
    /// </summary>
    public string MonitorMetric { get; set; } = "loss";

    /// <summary>
    /// Gets or sets the number of epochs with no improvement before stopping.
    /// </summary>
    public int Patience { get; set; } = 5;

    /// <summary>
    /// Gets or sets the minimum change to qualify as an improvement.
    /// </summary>
    public double MinDelta { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets whether lower values are better (true for loss, false for accuracy).
    /// </summary>
    public bool LowerIsBetter { get; set; } = true;
}
