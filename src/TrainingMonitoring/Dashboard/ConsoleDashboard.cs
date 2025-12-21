using System.Collections.Concurrent;
using System.Text;

namespace AiDotNet.TrainingMonitoring.Dashboard;

/// <summary>
/// Console-based training dashboard with ASCII charts.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> ConsoleDashboard provides real-time training visualization
/// directly in your terminal using ASCII art. This is useful when you don't have
/// access to a web browser or want a lightweight monitoring solution.
///
/// Features:
/// - ASCII line charts for loss/accuracy
/// - Progress bars for training progress
/// - Real-time metric display
/// - Colored output for different metric types
///
/// Example usage:
/// <code>
/// using var dashboard = new ConsoleDashboard();
/// dashboard.Start();
///
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     var (loss, accuracy) = TrainEpoch();
///     dashboard.LogScalar("loss", epoch, loss);
///     dashboard.LogScalar("accuracy", epoch, accuracy);
///     // Console updates automatically!
/// }
///
/// dashboard.Stop();
/// </code>
/// </remarks>
public class ConsoleDashboard : ITrainingDashboard
{
    private readonly ConcurrentDictionary<string, List<ScalarDataPoint>> _scalars = new();
    private readonly ConcurrentDictionary<string, List<HistogramDataPoint>> _histograms = new();
    private readonly object _renderLock = new();
    private Timer? _renderTimer;
    private bool _isRunning;
    private bool _disposed;
    private int _lastRenderHeight;

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public string LogDirectory { get; }

    /// <inheritdoc />
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Gets or sets the chart width in characters.
    /// </summary>
    public int ChartWidth { get; set; } = 60;

    /// <summary>
    /// Gets or sets the chart height in characters.
    /// </summary>
    public int ChartHeight { get; set; } = 10;

    /// <summary>
    /// Gets or sets the refresh interval in milliseconds.
    /// </summary>
    public int RefreshIntervalMs { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use colors.
    /// </summary>
    public bool UseColors { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to clear the screen on each render.
    /// </summary>
    public bool ClearOnRender { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of metrics to display.
    /// </summary>
    public int MaxMetricsDisplay { get; set; } = 6;

    /// <summary>
    /// Creates a new console dashboard.
    /// </summary>
    /// <param name="logDirectory">Directory to save logs.</param>
    /// <param name="name">Name of this training run.</param>
    public ConsoleDashboard(string? logDirectory = null, string? name = null)
    {
        LogDirectory = logDirectory ?? "./logs";
        Name = name ?? $"run_{DateTime.Now:yyyyMMdd_HHmmss}";
        Directory.CreateDirectory(LogDirectory);
    }

    /// <inheritdoc />
    public void Start()
    {
        if (_isRunning) return;

        _isRunning = true;
        _renderTimer = new Timer(_ => Render(), null, 0, RefreshIntervalMs);
    }

    /// <inheritdoc />
    public void Stop()
    {
        if (!_isRunning) return;

        _isRunning = false;
        _renderTimer?.Dispose();
        _renderTimer = null;

        // Final render
        Render();
    }

    /// <inheritdoc />
    public void LogScalar(string name, long step, double value, DateTime? wallTime = null)
    {
        var dataPoint = new ScalarDataPoint
        {
            Step = step,
            Value = value,
            WallTime = wallTime ?? DateTime.UtcNow
        };

        var series = _scalars.GetOrAdd(name, _ => new List<ScalarDataPoint>());
        lock (series)
        {
            series.Add(dataPoint);
        }
    }

    /// <inheritdoc />
    public void LogScalars(Dictionary<string, double> scalars, long step, DateTime? wallTime = null)
    {
        var time = wallTime ?? DateTime.UtcNow;
        foreach (var kvp in scalars)
        {
            LogScalar(kvp.Key, step, kvp.Value, time);
        }
    }

    /// <inheritdoc />
    public void LogHistogram(string name, long step, double[] values, DateTime? wallTime = null)
    {
        if (values.Length == 0) return;

        var sorted = values.OrderBy(v => v).ToArray();
        var dataPoint = new HistogramDataPoint
        {
            Step = step,
            WallTime = wallTime ?? DateTime.UtcNow,
            Min = sorted[0],
            Max = sorted[^1],
            Count = values.Length,
            Sum = sorted.Sum(),
            SumSquares = sorted.Sum(v => v * v)
        };

        var series = _histograms.GetOrAdd(name, _ => new List<HistogramDataPoint>());
        lock (series)
        {
            series.Add(dataPoint);
        }
    }

    /// <inheritdoc />
    public void LogImage(string name, long step, byte[] imageData, int width, int height, DateTime? wallTime = null)
    {
        // Images not supported in console - just log as text
        LogText($"{name}_image", step, $"[Image: {width}x{height}]", wallTime);
    }

    /// <inheritdoc />
    public void LogText(string name, long step, string text, DateTime? wallTime = null)
    {
        // For console, we just print important text
        lock (_renderLock)
        {
            if (UseColors) Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"[{name}] Step {step}: {text}");
            if (UseColors) Console.ResetColor();
        }
    }

    /// <inheritdoc />
    public void LogHyperparameters(Dictionary<string, object> hyperparams, Dictionary<string, double>? metrics = null)
    {
        lock (_renderLock)
        {
            if (UseColors) Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("=== Hyperparameters ===");
            foreach (var kvp in hyperparams)
            {
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            }
            if (UseColors) Console.ResetColor();
        }
    }

    /// <inheritdoc />
    public void LogConfusionMatrix(string name, long step, int[,] matrix, string[] labels, DateTime? wallTime = null)
    {
        // Display confusion matrix in console
        lock (_renderLock)
        {
            if (UseColors) Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"\n=== {name} (Step {step}) ===");

            var rows = matrix.GetLength(0);
            var cols = matrix.GetLength(1);

            // Header - use safe indexing for labels
            Console.Write("       ");
            for (int j = 0; j < cols; j++)
            {
                var colLabel = j < labels.Length ? labels[j] : $"C{j}";
                Console.Write($"{colLabel,8}");
            }
            Console.WriteLine();

            // Data rows - use safe indexing for labels
            for (int i = 0; i < rows; i++)
            {
                var rowLabel = i < labels.Length ? labels[i] : $"R{i}";
                Console.Write($"{rowLabel,6} ");
                for (int j = 0; j < cols; j++)
                {
                    Console.Write($"{matrix[i, j],8}");
                }
                Console.WriteLine();
            }

            if (UseColors) Console.ResetColor();
        }
    }

    /// <inheritdoc />
    public void LogPRCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null)
    {
        // Calculate and display AP
        var ap = CalculateAP(predictions, labels);
        lock (_renderLock)
        {
            if (UseColors) Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[{name}] Step {step}: Average Precision = {ap:F4}");
            if (UseColors) Console.ResetColor();
        }
    }

    /// <inheritdoc />
    public void LogROCCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null)
    {
        // Calculate and display AUC
        var auc = CalculateAUC(predictions, labels);
        lock (_renderLock)
        {
            if (UseColors) Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[{name}] Step {step}: AUC = {auc:F4}");
            if (UseColors) Console.ResetColor();
        }
    }

    /// <inheritdoc />
    public void LogModelGraph(string modelDescription)
    {
        lock (_renderLock)
        {
            if (UseColors) Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine("\n=== Model Architecture ===");
            Console.WriteLine(modelDescription);
            if (UseColors) Console.ResetColor();
        }
    }

    /// <inheritdoc />
    public string GenerateReport(string? outputPath = null)
    {
        var path = outputPath ?? Path.Combine(LogDirectory, $"{Name}_console_report.txt");
        var sb = new StringBuilder();

        sb.AppendLine($"Training Report: {Name}");
        sb.AppendLine($"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine(new string('=', 50));
        sb.AppendLine();

        foreach (var kvp in _scalars.OrderBy(k => k.Key))
        {
            if (kvp.Value.Count == 0) continue;

            sb.AppendLine($"Metric: {kvp.Key}");
            sb.AppendLine($"  Points: {kvp.Value.Count}");
            sb.AppendLine($"  Latest: {kvp.Value.Last().Value:F6}");
            sb.AppendLine($"  Min: {kvp.Value.Min(p => p.Value):F6}");
            sb.AppendLine($"  Max: {kvp.Value.Max(p => p.Value):F6}");
            sb.AppendLine();
        }

        File.WriteAllText(path, sb.ToString());
        return path;
    }

    /// <inheritdoc />
    public void ExportTensorBoardFormat(string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);
        var path = Path.Combine(outputDirectory, "scalars.csv");

        var sb = new StringBuilder();
        sb.AppendLine("metric,step,value,wall_time");

        foreach (var kvp in _scalars)
        {
            foreach (var point in kvp.Value)
            {
                sb.AppendLine($"{kvp.Key},{point.Step},{point.Value},{point.WallTime:O}");
            }
        }

        File.WriteAllText(path, sb.ToString());
    }

    /// <inheritdoc />
    public Dictionary<string, List<ScalarDataPoint>> GetScalarData()
    {
        return _scalars.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.ToList()
        );
    }

    /// <inheritdoc />
    public Dictionary<string, List<HistogramDataPoint>> GetHistogramData()
    {
        return _histograms.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.ToList()
        );
    }

    /// <inheritdoc />
    public void Clear()
    {
        _scalars.Clear();
        _histograms.Clear();
    }

    /// <inheritdoc />
    public void Flush()
    {
        Render();
    }

    private void Render()
    {
        if (!_isRunning && _scalars.Count == 0) return;

        lock (_renderLock)
        {
            var sb = new StringBuilder();

            // Header
            sb.AppendLine();
            sb.AppendLine($"  Training Dashboard: {Name}");
            sb.AppendLine($"  {DateTime.Now:HH:mm:ss} | Metrics: {_scalars.Count}");
            sb.AppendLine(new string('-', ChartWidth + 20));

            // Display metrics
            var metricsToShow = _scalars
                .OrderBy(k => k.Key)
                .Take(MaxMetricsDisplay)
                .ToList();

            foreach (var kvp in metricsToShow)
            {
                if (kvp.Value.Count == 0) continue;

                var latest = kvp.Value.Last();
                var data = kvp.Value.TakeLast(ChartWidth).Select(p => p.Value).ToArray();

                sb.AppendLine();
                sb.AppendLine($"  {kvp.Key}: {latest.Value:F6} (step {latest.Step})");

                // Write current buffer, then render sparkline directly to console with colors
                Console.Write(sb.ToString());
                sb.Clear();
                RenderSparklineToConsole(data);
            }

            // Summary table
            if (metricsToShow.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("  --- Summary ---");
                sb.AppendLine($"  {"Metric",-20} {"Current",12} {"Min",12} {"Max",12}");

                foreach (var kvp in metricsToShow)
                {
                    if (kvp.Value.Count == 0) continue;

                    var current = kvp.Value.Last().Value;
                    var min = kvp.Value.Min(p => p.Value);
                    var max = kvp.Value.Max(p => p.Value);

                    sb.AppendLine($"  {Truncate(kvp.Key, 20),-20} {current,12:F6} {min,12:F6} {max,12:F6}");
                }
            }

            sb.AppendLine(new string('-', ChartWidth + 20));

            if (ClearOnRender)
            {
                // Move cursor up and clear previous output
                try
                {
                    var lines = sb.ToString().Split('\n').Length;
                    if (_lastRenderHeight > 0)
                    {
                        Console.SetCursorPosition(0, Math.Max(0, Console.CursorTop - _lastRenderHeight));
                        for (int i = 0; i < _lastRenderHeight; i++)
                        {
                            Console.Write(new string(' ', Console.WindowWidth));
                        }
                        Console.SetCursorPosition(0, Math.Max(0, Console.CursorTop - _lastRenderHeight));
                    }
                    _lastRenderHeight = lines;
                }
                catch
                {
                    // Cursor operations might fail in some terminals
                }
            }

            Console.Write(sb.ToString());
        }
    }

    private void RenderSparklineToConsole(double[] data)
    {
        if (data.Length == 0) return;

        var min = data.Min();
        var max = data.Max();
        var range = max - min;

        // Use block characters for better resolution
        var blocks = new[] { ' ', '\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588' };

        Console.Write("  ");

        foreach (var value in data)
        {
            int blockIndex;
            if (range <= 0)
            {
                blockIndex = 4; // Middle block if all values are same
            }
            else
            {
                blockIndex = (int)((value - min) / range * 7);
                blockIndex = Math.Max(0, Math.Min(7, blockIndex));
            }

            if (UseColors)
            {
                // Color based on position in range
                var normalized = range > 0 ? (value - min) / range : 0.5;
                if (normalized < 0.33)
                    Console.ForegroundColor = ConsoleColor.Green;
                else if (normalized < 0.66)
                    Console.ForegroundColor = ConsoleColor.Yellow;
                else
                    Console.ForegroundColor = ConsoleColor.Red;
            }

            Console.Write(blocks[blockIndex]);

            if (UseColors)
                Console.ResetColor();
        }

        Console.WriteLine();
    }

    private static string Truncate(string text, int maxLength)
    {
        if (text.Length <= maxLength) return text;
        return text.Substring(0, maxLength - 3) + "...";
    }

    private static double CalculateAP(double[] predictions, int[] labels)
    {
        var sorted = predictions.Zip(labels, (p, l) => (pred: p, label: l))
            .OrderByDescending(x => x.pred)
            .ToList();

        int totalPositives = labels.Sum();
        if (totalPositives == 0) return 0;

        int tp = 0;
        double ap = 0;

        for (int i = 0; i < sorted.Count; i++)
        {
            if (sorted[i].label == 1)
            {
                tp++;
                var precision = (double)tp / (i + 1);
                ap += precision;
            }
        }

        return ap / totalPositives;
    }

    private static double CalculateAUC(double[] predictions, int[] labels)
    {
        var sorted = predictions.Zip(labels, (p, l) => (pred: p, label: l))
            .OrderByDescending(x => x.pred)
            .ToList();

        int totalPositives = labels.Sum();
        int totalNegatives = labels.Length - totalPositives;
        if (totalPositives == 0 || totalNegatives == 0) return 0.5;

        int tp = 0, fp = 0;
        double auc = 0;
        double prevFpr = 0, prevTpr = 0;

        for (int i = 0; i < sorted.Count; i++)
        {
            if (sorted[i].label == 1) tp++;
            else fp++;

            var tpr = (double)tp / totalPositives;
            var fpr = (double)fp / totalNegatives;

            // Trapezoidal integration
            auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;

            prevFpr = fpr;
            prevTpr = tpr;
        }

        return auc;
    }

    /// <summary>
    /// Disposes the console dashboard.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        Stop();
    }
}
