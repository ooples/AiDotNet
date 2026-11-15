using System.Diagnostics;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for automatic checkpointing during training, supporting step-frequency,
/// monotonic wall-clock interval, and metric-improvement triggers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Automatic checkpointing saves your model's training progress periodically
/// so you never lose hours of work if a machine restarts or a long run is interrupted. You can trigger saves:
/// - Every N optimizer steps (<see cref="SaveFrequency"/>),
/// - Every wall-clock time interval such as every 15 minutes (<see cref="SaveInterval"/>),
/// - Whenever a validation or training metric improves (<see cref="SaveOnImprovement"/>),
/// or any combination of the three.</para>
/// </remarks>
public sealed class AutoCheckpointingOptions : ModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="AutoCheckpointingOptions"/> class with default values.
    /// </summary>
    public AutoCheckpointingOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="AutoCheckpointingOptions"/> class by copying another instance.
    /// </summary>
    /// <param name="other">The options instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public AutoCheckpointingOptions(AutoCheckpointingOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        SaveFrequency = other.SaveFrequency;
        KeepLast = other.KeepLast;
        SaveOnImprovement = other.SaveOnImprovement;
        MetricName = other.MetricName;
        MinimizeMetric = other.MinimizeMetric;
        SaveInterval = other.SaveInterval;
    }

    /// <summary>
    /// Gets or sets the number of steps between automatic checkpoint saves. Set to <c>0</c> to disable
    /// step-frequency saving (for example, when relying purely on <see cref="SaveInterval"/> or
    /// <see cref="SaveOnImprovement"/>).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Setting this to <c>100</c> saves a checkpoint every 100 training steps.
    /// Set it to <c>0</c> if you only want time-based or best-metric checkpoints.</para>
    /// </remarks>
    public int SaveFrequency { get; set; }

    /// <summary>
    /// Gets or sets the number of most recent checkpoints to retain when cleaning up older checkpoints.
    /// Defaults to <c>5</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Checkpoints can be large on disk. Keeping the last 5 saves protects you
    /// from recent issues without filling up your drive.</para>
    /// </remarks>
    public int KeepLast { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to save a checkpoint whenever the tracked metric improves. Defaults to <c>true</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, a checkpoint is written each time your model achieves a new
    /// best score (lowest loss or highest accuracy).</para>
    /// </remarks>
    public bool SaveOnImprovement { get; set; } = true;

    /// <summary>
    /// Gets or sets the name of the metric tracked when <see cref="SaveOnImprovement"/> is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For example, set this to <c>"val_loss"</c> or <c>"loss"</c> to track
    /// error, or <c>"accuracy"</c> (with <see cref="MinimizeMetric"/> set to <c>false</c>) to track accuracy.</para>
    /// </remarks>
    public string? MetricName { get; set; }

    /// <summary>
    /// Gets or sets whether lower values of <see cref="MetricName"/> represent improvement (<c>true</c>) or
    /// higher values represent improvement (<c>false</c>). Defaults to <c>true</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keep this <c>true</c> for loss/error metrics where smaller is better, and
    /// set it to <c>false</c> for accuracy/F1 metrics where larger is better.</para>
    /// </remarks>
    public bool MinimizeMetric { get; set; } = true;

    /// <summary>
    /// Gets or sets an optional monotonic wall-clock interval (measured via <see cref="Stopwatch.GetTimestamp"/>)
    /// after which the next completed training step triggers an automatic checkpoint.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set this to <c>TimeSpan.FromMinutes(15)</c> to guarantee a checkpoint is
    /// saved at least every 15 minutes regardless of how fast or slow each step is.</para>
    /// </remarks>
    public TimeSpan? SaveInterval { get; set; }

    /// <summary>
    /// Validates the configured auto-checkpointing options.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a numeric or interval value is out of range.</exception>
    public void Validate()
    {
        if (SaveFrequency < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(SaveFrequency), "Save frequency cannot be negative.");
        }

        if (KeepLast < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(KeepLast), "KeepLast cannot be negative.");
        }

        if (SaveInterval is { TotalMilliseconds: < 0 })
        {
            throw new ArgumentOutOfRangeException(nameof(SaveInterval), "Save interval cannot be negative.");
        }
    }
}
