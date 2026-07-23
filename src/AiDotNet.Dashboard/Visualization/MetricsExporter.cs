using Newtonsoft.Json;

namespace AiDotNet.Dashboard.Visualization;

/// <summary>
/// Exports training metrics to various formats for external visualization tools.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This class exports your training metrics so you can visualize them
/// in external tools like TensorBoard, MLflow, or custom dashboards.
///
/// Supported formats:
/// - CSV: Simple spreadsheet format for Excel, pandas, etc.
/// - JSON: Structured data for web dashboards
/// - TensorBoard: Event files compatible with TensorBoard
///
/// Example usage:
/// <code>
/// var exporter = new MetricsExporter("./logs/run-001");
///
/// // Log metrics during training
/// exporter.LogScalar("train/loss", 0.5, epoch);
/// exporter.LogScalar("train/accuracy", 0.85, epoch);
///
/// // Export to different formats
/// exporter.ExportToCsv("metrics.csv");
/// exporter.ExportToJson("metrics.json");
/// </code>
/// </remarks>
public class MetricsExporter : IDisposable
{
    private readonly string _outputDirectory;
    private readonly Dictionary<string, List<MetricEntry>> _metrics;
    private readonly object _lock = new();
    private bool _isDisposed;

    /// <summary>
    /// Gets or sets whether to auto-flush metrics to disk.
    /// </summary>
    public bool AutoFlush { get; set; } = true;

    /// <summary>
    /// Gets or sets the auto-flush interval in entries.
    /// </summary>
    public int FlushInterval { get; set; } = 100;

    private int _entryCount;

    /// <summary>
    /// Initializes a new instance of the MetricsExporter class.
    /// </summary>
    /// <param name="outputDirectory">Directory to store exported metrics.</param>
    /// <exception cref="ArgumentException">Thrown when the output directory path is invalid or contains path traversal sequences.</exception>
    public MetricsExporter(string? outputDirectory = null)
    {
        var rawPath = outputDirectory ?? Path.Combine(".", "metrics_export");
        _outputDirectory = ValidateAndSanitizeDirectory(rawPath);
        _metrics = new Dictionary<string, List<MetricEntry>>();

        if (!Directory.Exists(_outputDirectory))
        {
            Directory.CreateDirectory(_outputDirectory);
        }
    }

    /// <summary>
    /// Validates and sanitizes a directory path to prevent path traversal attacks.
    /// </summary>
    private static string ValidateAndSanitizeDirectory(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentException("Output directory path cannot be null or empty.", nameof(path));
        }

        // Check for path traversal sequences
        if (path.Contains(".."))
        {
            throw new ArgumentException("Output directory path cannot contain path traversal sequences (..).", nameof(path));
        }

        // Get the full path and ensure it's properly resolved
        var fullPath = Path.GetFullPath(path);

        // Ensure the path doesn't escape to sensitive system directories
        var normalizedPath = fullPath.Replace('\\', '/').ToLowerInvariant();
        var sensitiveRoots = new[] { "/windows", "/system32", "/etc", "/var", "/usr", "/bin", "/sbin", "/root" };
        foreach (var root in sensitiveRoots)
        {
            if (normalizedPath.StartsWith(root, StringComparison.OrdinalIgnoreCase) ||
                normalizedPath.Contains($":{root}", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"Output directory cannot be in a sensitive system directory: {path}", nameof(path));
            }
        }

        return fullPath;
    }

    /// <summary>
    /// Validates and sanitizes a file name to prevent path traversal attacks.
    /// </summary>
    private string ValidateAndSanitizeFileName(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
        {
            throw new ArgumentException("File name cannot be null or empty.", nameof(fileName));
        }

        // Strip any directory components - only allow file name
        var sanitizedName = Path.GetFileName(fileName);

        if (string.IsNullOrWhiteSpace(sanitizedName))
        {
            throw new ArgumentException("File name is invalid after sanitization.", nameof(fileName));
        }

        // Check for path traversal sequences
        if (sanitizedName.Contains("..") || sanitizedName.Contains('/') || sanitizedName.Contains('\\'))
        {
            throw new ArgumentException("File name cannot contain path traversal sequences.", nameof(fileName));
        }

        // Verify the combined path stays within the output directory
        var fullPath = Path.GetFullPath(Path.Combine(_outputDirectory, sanitizedName));
        var outputDirFull = Path.GetFullPath(_outputDirectory);

        if (!fullPath.StartsWith(outputDirFull, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("File path escapes the output directory.", nameof(fileName));
        }

        return sanitizedName;
    }

    /// <summary>
    /// Logs a scalar metric value.
    /// </summary>
    /// <param name="tag">The metric tag (e.g., "train/loss").</param>
    /// <param name="value">The metric value.</param>
    /// <param name="step">The global step or epoch.</param>
    /// <param name="wallTime">Optional wall clock time (defaults to now).</param>
    public void LogScalar(string tag, double value, long step, DateTime? wallTime = null)
    {
        if (string.IsNullOrWhiteSpace(tag))
            throw new ArgumentException("Tag cannot be null or empty.", nameof(tag));

        lock (_lock)
        {
            if (!_metrics.ContainsKey(tag))
            {
                _metrics[tag] = new List<MetricEntry>();
            }

            _metrics[tag].Add(new MetricEntry
            {
                Tag = tag,
                Value = value,
                Step = step,
                WallTime = wallTime ?? DateTime.UtcNow
            });

            _entryCount++;

            if (AutoFlush && _entryCount >= FlushInterval)
            {
                Flush();
                _entryCount = 0;
            }
        }
    }

    /// <summary>
    /// Logs multiple scalar metrics at once.
    /// </summary>
    /// <param name="metrics">Dictionary of tag -> value pairs.</param>
    /// <param name="step">The global step or epoch.</param>
    public void LogScalars(Dictionary<string, double> metrics, long step)
    {
        if (metrics == null)
            return;

        var wallTime = DateTime.UtcNow;
        foreach (var kvp in metrics)
        {
            LogScalar(kvp.Key, kvp.Value, step, wallTime);
        }
    }

    /// <summary>
    /// Logs a histogram of values.
    /// </summary>
    /// <param name="tag">The histogram tag.</param>
    /// <param name="values">The values to histogram.</param>
    /// <param name="step">The global step.</param>
    public void LogHistogram(string tag, double[] values, long step)
    {
        if (values == null || values.Length == 0)
            return;

        lock (_lock)
        {
            // Store histogram summary as multiple scalars
            LogScalar($"{tag}/mean", values.Average(), step);
            LogScalar($"{tag}/std", CalculateStd(values), step);
            LogScalar($"{tag}/min", values.Min(), step);
            LogScalar($"{tag}/max", values.Max(), step);
            LogScalar($"{tag}/count", values.Length, step);
        }
    }

    /// <summary>
    /// Logs hyperparameters.
    /// </summary>
    /// <param name="hyperparams">Dictionary of hyperparameter names and values.</param>
    public void LogHyperparameters(Dictionary<string, object> hyperparams)
    {
        if (hyperparams == null)
            return;

        lock (_lock)
        {
            var path = Path.Combine(_outputDirectory, "hyperparameters.json");
            var json = JsonConvert.SerializeObject(hyperparams, Formatting.Indented);
            File.WriteAllText(path, json);
        }
    }

    /// <summary>
    /// Exports all metrics to CSV format.
    /// </summary>
    /// <param name="fileName">Output file name (relative to output directory).</param>
    /// <exception cref="ArgumentException">Thrown when the file name contains path traversal sequences.</exception>
    public void ExportToCsv(string fileName = "metrics.csv")
    {
        var sanitizedFileName = ValidateAndSanitizeFileName(fileName);
        lock (_lock)
        {
            var path = Path.Combine(_outputDirectory, sanitizedFileName);
            var lines = new List<string> { "tag,step,value,wall_time" };

            foreach (var kvp in _metrics)
            {
                foreach (var entry in kvp.Value)
                {
                    var escapedTag = EscapeCsvField(entry.Tag);
                    // Use InvariantCulture to ensure consistent decimal separator (.) across all locales
                    var valueStr = entry.Value.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    lines.Add($"{escapedTag},{entry.Step},{valueStr},{entry.WallTime:o}");
                }
            }

            File.WriteAllLines(path, lines);
        }
    }

    /// <summary>
    /// Escapes a string value for safe inclusion in a CSV field.
    /// </summary>
    private static string EscapeCsvField(string value)
    {
        if (string.IsNullOrEmpty(value))
            return value;

        // If the value contains special characters, wrap in quotes and escape internal quotes
        if (value.Contains(',') || value.Contains('"') || value.Contains('\n') || value.Contains('\r'))
        {
            return $"\"{value.Replace("\"", "\"\"")}\"";
        }

        return value;
    }

    /// <summary>
    /// Exports all metrics to JSON format.
    /// </summary>
    /// <param name="fileName">Output file name (relative to output directory).</param>
    /// <exception cref="ArgumentException">Thrown when the file name contains path traversal sequences.</exception>
    public void ExportToJson(string fileName = "metrics.json")
    {
        var sanitizedFileName = ValidateAndSanitizeFileName(fileName);
        lock (_lock)
        {
            var path = Path.Combine(_outputDirectory, sanitizedFileName);
            var export = new MetricsExport
            {
                ExportedAt = DateTime.UtcNow,
                Metrics = _metrics.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value.Select(e => new MetricPoint
                    {
                        Step = e.Step,
                        Value = e.Value,
                        WallTime = e.WallTime
                    }).ToList()
                )
            };

            var json = JsonConvert.SerializeObject(export, Formatting.Indented);
            File.WriteAllText(path, json);
        }
    }

    /// <summary>
    /// Exports metrics in a format compatible with TensorBoard's event files.
    /// </summary>
    /// <remarks>
    /// This creates a simplified JSON representation that can be converted
    /// to TensorBoard events using Python tools.
    /// </remarks>
    /// <param name="fileName">Output file name.</param>
    /// <exception cref="ArgumentException">Thrown when the file name contains path traversal sequences.</exception>
    public void ExportToTensorBoardFormat(string fileName = "tensorboard_events.json")
    {
        var sanitizedFileName = ValidateAndSanitizeFileName(fileName);
        lock (_lock)
        {
            var path = Path.Combine(_outputDirectory, sanitizedFileName);
            var events = new List<TensorBoardEvent>();

            foreach (var kvp in _metrics)
            {
                foreach (var entry in kvp.Value)
                {
                    events.Add(new TensorBoardEvent
                    {
                        WallTime = new DateTimeOffset(entry.WallTime).ToUnixTimeSeconds() +
                                   entry.WallTime.Millisecond / 1000.0,
                        Step = entry.Step,
                        Tag = entry.Tag,
                        SimpleValue = entry.Value
                    });
                }
            }

            var json = JsonConvert.SerializeObject(events, Formatting.Indented);
            File.WriteAllText(path, json);
        }
    }

    /// <summary>
    /// Exports a summary report.
    /// </summary>
    /// <param name="fileName">Output file name.</param>
    /// <exception cref="ArgumentException">Thrown when the file name contains path traversal sequences.</exception>
    public void ExportSummaryReport(string fileName = "summary.txt")
    {
        var sanitizedFileName = ValidateAndSanitizeFileName(fileName);
        lock (_lock)
        {
            var path = Path.Combine(_outputDirectory, sanitizedFileName);
            var lines = new List<string>
            {
                "Training Metrics Summary",
                "========================",
                $"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC",
                ""
            };

            foreach (var kvp in _metrics.OrderBy(k => k.Key))
            {
                var values = kvp.Value.Select(e => e.Value).ToList();
                if (values.Count == 0) continue;

                lines.Add($"{kvp.Key}:");
                lines.Add($"  Count: {values.Count}");
                lines.Add($"  Min: {values.Min():F6}");
                lines.Add($"  Max: {values.Max():F6}");
                lines.Add($"  Mean: {values.Average():F6}");
                lines.Add($"  Final: {values.Last():F6}");
                lines.Add("");
            }

            File.WriteAllLines(path, lines);
        }
    }

    /// <summary>
    /// Flushes any buffered metrics to disk.
    /// </summary>
    public void Flush()
    {
        // Export all current metrics
        ExportToJson();
    }

    /// <summary>
    /// Clears all stored metrics.
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _metrics.Clear();
            _entryCount = 0;
        }
    }

    /// <summary>
    /// Gets all entries for a specific metric.
    /// </summary>
    public List<(long step, double value)> GetMetric(string tag)
    {
        lock (_lock)
        {
            if (_metrics.TryGetValue(tag, out var entries))
            {
                return entries.Select(e => (e.Step, e.Value)).ToList();
            }
            return new List<(long, double)>();
        }
    }

    /// <summary>
    /// Gets all available metric tags.
    /// </summary>
    public List<string> GetAllTags()
    {
        lock (_lock)
        {
            return _metrics.Keys.OrderBy(k => k).ToList();
        }
    }

    private static double CalculateStd(double[] values)
    {
        if (values.Length < 2) return 0;
        var mean = values.Average();
        var sumSquaredDiff = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSquaredDiff / (values.Length - 1));
    }

    /// <summary>
    /// Disposes the exporter and flushes remaining metrics.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
            return;

        _isDisposed = true;
        Flush();
        ExportSummaryReport();
    }

    #region Nested Types

    private class MetricEntry
    {
        public string Tag { get; set; } = string.Empty;
        public double Value { get; set; }
        public long Step { get; set; }
        public DateTime WallTime { get; set; }
    }

    private class MetricsExport
    {
        public DateTime ExportedAt { get; set; }
        public Dictionary<string, List<MetricPoint>> Metrics { get; set; } = new();
    }

    private class MetricPoint
    {
        public long Step { get; set; }
        public double Value { get; set; }
        public DateTime WallTime { get; set; }
    }

    private class TensorBoardEvent
    {
        [JsonProperty("wall_time")]
        public double WallTime { get; set; }

        [JsonProperty("step")]
        public long Step { get; set; }

        [JsonProperty("tag")]
        public string Tag { get; set; } = string.Empty;

        [JsonProperty("simple_value")]
        public double SimpleValue { get; set; }
    }

    #endregion
}
