namespace AiDotNet.Dashboard.Console;

/// <summary>
/// A rich console progress bar for training visualization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This progress bar provides visual feedback during training:
/// - Shows completion percentage with animated bar
/// - Displays elapsed time and estimated time remaining
/// - Shows current metrics (loss, accuracy, etc.)
/// - Supports nested progress for epoch/batch tracking
///
/// Example usage:
/// <code>
/// using var progress = new ProgressBar(totalEpochs, "Training");
/// for (int epoch = 0; epoch &lt; totalEpochs; epoch++)
/// {
///     progress.Update(epoch + 1);
///     progress.SetMetric("loss", epochLoss);
///     progress.SetMetric("acc", accuracy);
/// }
/// </code>
/// </remarks>
public class ProgressBar : IDisposable
{
    private readonly int _total;
    private readonly string _description;
    private readonly DateTime _startTime;
    private readonly object _lock = new();
    private readonly Dictionary<string, double> _metrics;
    private readonly bool _useColors;

    private int _current;
    private int _barWidth;
    private bool _isDisposed;
    private string _status;
    private ProgressBar? _childProgress;

    /// <summary>
    /// Gets or sets whether to show the progress bar (can be disabled for non-interactive environments).
    /// </summary>
    public bool IsVisible { get; set; } = true;

    /// <summary>
    /// Gets the current progress value.
    /// </summary>
    public int Current => _current;

    /// <summary>
    /// Gets the total value for completion.
    /// </summary>
    public int Total => _total;

    /// <summary>
    /// Gets the progress percentage (0-100).
    /// </summary>
    public double Percentage => _total > 0 ? (_current * 100.0 / _total) : 0;

    /// <summary>
    /// Gets the elapsed time since progress started.
    /// </summary>
    public TimeSpan Elapsed => DateTime.UtcNow - _startTime;

    /// <summary>
    /// Initializes a new instance of the ProgressBar class.
    /// </summary>
    /// <param name="total">The total number of steps.</param>
    /// <param name="description">A description of the operation.</param>
    /// <param name="barWidth">Width of the progress bar in characters.</param>
    /// <param name="useColors">Whether to use console colors.</param>
    public ProgressBar(int total, string description = "Progress", int barWidth = 40, bool useColors = true)
    {
        if (total < 0)
            throw new ArgumentException("Total must be non-negative.", nameof(total));

        _total = total;
        _description = description;
        _barWidth = barWidth;
        _useColors = useColors;
        _startTime = DateTime.UtcNow;
        _current = 0;
        _metrics = new Dictionary<string, double>();
        _status = string.Empty;
    }

    /// <summary>
    /// Updates the progress to a specific value.
    /// </summary>
    /// <param name="current">The current progress value.</param>
    public void Update(int current)
    {
        lock (_lock)
        {
            _current = Math.Min(current, _total);
            Render();
        }
    }

    /// <summary>
    /// Increments the progress by one step.
    /// </summary>
    public void Increment()
    {
        lock (_lock)
        {
            _current = Math.Min(_current + 1, _total);
            Render();
        }
    }

    /// <summary>
    /// Sets a metric value to display alongside the progress bar.
    /// </summary>
    /// <param name="name">The metric name.</param>
    /// <param name="value">The metric value.</param>
    public void SetMetric(string name, double value)
    {
        lock (_lock)
        {
            _metrics[name] = value;
            Render();
        }
    }

    /// <summary>
    /// Sets multiple metrics at once.
    /// </summary>
    /// <param name="metrics">Dictionary of metric names and values.</param>
    public void SetMetrics(Dictionary<string, double> metrics)
    {
        if (metrics == null)
            return;

        lock (_lock)
        {
            foreach (var kvp in metrics)
            {
                _metrics[kvp.Key] = kvp.Value;
            }
            Render();
        }
    }

    /// <summary>
    /// Sets a status message to display.
    /// </summary>
    /// <param name="status">The status message.</param>
    public void SetStatus(string status)
    {
        lock (_lock)
        {
            _status = status ?? string.Empty;
            Render();
        }
    }

    /// <summary>
    /// Creates a child progress bar for nested operations (e.g., batches within epochs).
    /// </summary>
    /// <param name="total">Total steps for child progress.</param>
    /// <param name="description">Description for child progress.</param>
    /// <returns>A new child ProgressBar.</returns>
    public ProgressBar CreateChild(int total, string description = "Batch")
    {
        _childProgress = new ProgressBar(total, description, _barWidth / 2, _useColors)
        {
            IsVisible = IsVisible
        };
        return _childProgress;
    }

    /// <summary>
    /// Clears the child progress bar.
    /// </summary>
    public void ClearChild()
    {
        if (_childProgress != null)
        {
            _childProgress.Dispose();
            _childProgress = null;
            Render();
        }
    }

    /// <summary>
    /// Renders the progress bar to the console.
    /// </summary>
    private void Render()
    {
        if (!IsVisible || _isDisposed)
            return;

        try
        {
            var output = BuildProgressString();

            // Move cursor to beginning of line and clear
            // Console.WindowWidth can throw in non-interactive environments
            int windowWidth;
            try
            {
                windowWidth = System.Console.WindowWidth;
            }
            catch (IOException)
            {
                // Non-interactive console (e.g., redirected output)
                windowWidth = 120; // Use reasonable default
            }

            System.Console.Write("\r");
            System.Console.Write(new string(' ', Math.Max(0, windowWidth - 1)));
            System.Console.Write("\r");

            // Write progress
            if (_useColors)
            {
                WriteColored(output);
            }
            else
            {
                System.Console.Write(output);
            }
        }
        catch (IOException)
        {
            // Console might not be available
            IsVisible = false;
        }
    }

    /// <summary>
    /// Builds the progress string for display.
    /// </summary>
    private string BuildProgressString()
    {
        var percentage = Percentage;
        var filledWidth = (int)(percentage / 100.0 * _barWidth);
        var emptyWidth = _barWidth - filledWidth;

        var bar = $"[{new string('=', filledWidth)}{(filledWidth < _barWidth ? ">" : "")}{new string(' ', Math.Max(0, emptyWidth - 1))}]";

        var elapsed = Elapsed;
        var eta = GetEstimatedTimeRemaining();

        var timeStr = eta.HasValue && eta.Value.TotalSeconds > 0
            ? $"{FormatTime(elapsed)} < {FormatTime(eta.Value)}"
            : FormatTime(elapsed);

        var metricsStr = string.Join(" | ", _metrics.Select(m => $"{m.Key}: {m.Value:F4}"));

        var result = $"{_description}: {percentage,5:F1}% {bar} {_current}/{_total} [{timeStr}]";

        if (!string.IsNullOrEmpty(metricsStr))
        {
            result += $" | {metricsStr}";
        }

        if (!string.IsNullOrEmpty(_status))
        {
            result += $" | {_status}";
        }

        return result;
    }

    /// <summary>
    /// Writes colored output to console.
    /// </summary>
    private void WriteColored(string output)
    {
        var originalColor = System.Console.ForegroundColor;

        // Color based on completion
        if (_current >= _total)
        {
            System.Console.ForegroundColor = ConsoleColor.Green;
        }
        else if (Percentage > 50)
        {
            System.Console.ForegroundColor = ConsoleColor.Yellow;
        }
        else
        {
            System.Console.ForegroundColor = ConsoleColor.Cyan;
        }

        System.Console.Write(output);
        System.Console.ForegroundColor = originalColor;
    }

    /// <summary>
    /// Gets the estimated time remaining.
    /// </summary>
    private TimeSpan? GetEstimatedTimeRemaining()
    {
        if (_current <= 0)
            return null;

        var elapsed = Elapsed;
        var avgTimePerItem = elapsed.TotalSeconds / _current;
        var remaining = (_total - _current) * avgTimePerItem;

        return TimeSpan.FromSeconds(remaining);
    }

    /// <summary>
    /// Formats a TimeSpan for display.
    /// </summary>
    private static string FormatTime(TimeSpan time)
    {
        if (time.TotalHours >= 1)
        {
            return $"{(int)time.TotalHours}:{time.Minutes:D2}:{time.Seconds:D2}";
        }
        else if (time.TotalMinutes >= 1)
        {
            return $"{time.Minutes}:{time.Seconds:D2}";
        }
        else
        {
            return $"{time.Seconds}.{time.Milliseconds / 100}s";
        }
    }

    /// <summary>
    /// Completes the progress bar and moves to a new line.
    /// </summary>
    public void Complete()
    {
        lock (_lock)
        {
            _current = _total;
            Render();
            if (IsVisible)
            {
                System.Console.WriteLine();
            }
        }
    }

    /// <summary>
    /// Disposes the progress bar.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
        _childProgress?.Dispose();

        if (IsVisible && _current < _total)
        {
            // Incomplete progress - move to new line
            try
            {
                System.Console.WriteLine();
            }
            catch (IOException)
            {
                // Ignore
            }
        }
    }
}

/// <summary>
/// A multi-progress display for tracking multiple concurrent operations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Use this when you have multiple things to track at once:
/// - Multiple training runs
/// - Parallel hyperparameter trials
/// - Distributed training across nodes
/// </remarks>
public class MultiProgressDisplay : IDisposable
{
    private readonly List<ProgressBar> _progressBars;
    private readonly object _lock = new();
    private bool _isDisposed;

    /// <summary>
    /// Initializes a new instance of the MultiProgressDisplay class.
    /// </summary>
    public MultiProgressDisplay()
    {
        _progressBars = new List<ProgressBar>();
    }

    /// <summary>
    /// Adds a new progress bar to track.
    /// </summary>
    /// <param name="total">Total steps for the progress bar.</param>
    /// <param name="description">Description of the operation.</param>
    /// <returns>The created ProgressBar.</returns>
    public ProgressBar AddProgressBar(int total, string description)
    {
        lock (_lock)
        {
            var progress = new ProgressBar(total, description);
            _progressBars.Add(progress);
            return progress;
        }
    }

    /// <summary>
    /// Removes a progress bar from tracking.
    /// </summary>
    /// <param name="progressBar">The progress bar to remove.</param>
    public void RemoveProgressBar(ProgressBar progressBar)
    {
        lock (_lock)
        {
            _progressBars.Remove(progressBar);
            progressBar.Dispose();
        }
    }

    /// <summary>
    /// Clears all progress bars.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            foreach (var pb in _progressBars)
            {
                pb.Dispose();
            }
            _progressBars.Clear();
        }
    }

    /// <summary>
    /// Disposes all progress bars.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
        Clear();
    }
}
