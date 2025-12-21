#if !NET6_0_OR_GREATER
#pragma warning disable CS8600, CS8601, CS8602, CS8603, CS8604
#endif
using System.Collections.Concurrent;
using System.Globalization;
using System.Text;
using Newtonsoft.Json;

namespace AiDotNet.TrainingMonitoring.Dashboard;

/// <summary>
/// Generates interactive HTML dashboards for training visualization.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> HtmlDashboard creates beautiful, interactive HTML reports
/// that show your training progress. Unlike TensorBoard which requires running a server,
/// these HTML files can be opened directly in any web browser.
///
/// Features:
/// - Interactive charts with zoom and pan (using Chart.js)
/// - Real-time loss and accuracy curves
/// - Confusion matrix heatmaps
/// - Histogram distributions
/// - ROC and PR curves
/// - Hyperparameter logging
///
/// Example usage:
/// <code>
/// var dashboard = new HtmlDashboard("./logs", "my_experiment");
///
/// // During training
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     var (loss, accuracy) = TrainEpoch();
///     dashboard.LogScalar("train/loss", epoch, loss);
///     dashboard.LogScalar("train/accuracy", epoch, accuracy);
/// }
///
/// // Generate report
/// string reportPath = dashboard.GenerateReport();
/// Console.WriteLine($"Report saved to: {reportPath}");
/// </code>
/// </remarks>
public class HtmlDashboard : ITrainingDashboard
{
    private readonly ConcurrentDictionary<string, List<ScalarDataPoint>> _scalars = new();
    private readonly ConcurrentDictionary<string, List<HistogramDataPoint>> _histograms = new();
    private readonly ConcurrentDictionary<string, List<ImageDataPoint>> _images = new();
    private readonly ConcurrentDictionary<string, List<TextDataPoint>> _texts = new();
    private readonly ConcurrentDictionary<string, List<ConfusionMatrixDataPoint>> _confusionMatrices = new();
    private readonly ConcurrentDictionary<string, List<CurveDataPoint>> _prCurves = new();
    private readonly ConcurrentDictionary<string, List<CurveDataPoint>> _rocCurves = new();
    private readonly List<Dictionary<string, object>> _hyperparameters = new();
    private readonly object _lock = new();
    private string? _modelGraph;
    private bool _isRunning;
    private bool _disposed;

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public string LogDirectory { get; }

    /// <inheritdoc />
    public bool IsRunning => _isRunning;

    /// <summary>
    /// Gets or sets whether to auto-save on each log operation.
    /// </summary>
    public bool AutoSave { get; set; } = false;

    /// <summary>
    /// Gets or sets the auto-save interval in number of logs.
    /// </summary>
    public int AutoSaveInterval { get; set; } = 100;

    private int _logCount = 0;

    /// <summary>
    /// Creates a new HTML dashboard.
    /// </summary>
    /// <param name="logDirectory">Directory to save logs and reports.</param>
    /// <param name="name">Name of this training run.</param>
    public HtmlDashboard(string logDirectory, string? name = null)
    {
        LogDirectory = logDirectory;
        Name = name ?? $"run_{DateTime.Now:yyyyMMdd_HHmmss}";

        Directory.CreateDirectory(logDirectory);
    }

    /// <inheritdoc />
    public void Start()
    {
        _isRunning = true;
    }

    /// <inheritdoc />
    public void Stop()
    {
        _isRunning = false;
        Flush();
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

        CheckAutoSave();
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
        var min = sorted[0];
        var max = sorted[^1];
        var sum = sorted.Sum();
        var sumSquares = sorted.Sum(v => v * v);

        // Create histogram buckets
        const int numBuckets = 30;
        var bucketLimits = new double[numBuckets];
        var bucketCounts = new int[numBuckets];
        var range = max - min;

        if (range <= 0)
        {
            bucketLimits[0] = min;
            bucketCounts[0] = values.Length;
        }
        else
        {
            for (int i = 0; i < numBuckets; i++)
            {
                bucketLimits[i] = min + ((i + 1) * range / numBuckets);
            }

            foreach (var v in values)
            {
                var bucketIndex = (int)((v - min) / range * (numBuckets - 1));
                bucketIndex = Math.Max(0, Math.Min(numBuckets - 1, bucketIndex));
                bucketCounts[bucketIndex]++;
            }
        }

        var dataPoint = new HistogramDataPoint
        {
            Step = step,
            WallTime = wallTime ?? DateTime.UtcNow,
            Min = min,
            Max = max,
            Count = values.Length,
            Sum = sum,
            SumSquares = sumSquares,
            BucketLimits = bucketLimits,
            BucketCounts = bucketCounts
        };

        var series = _histograms.GetOrAdd(name, _ => new List<HistogramDataPoint>());
        lock (series)
        {
            series.Add(dataPoint);
        }

        CheckAutoSave();
    }

    /// <inheritdoc />
    public void LogImage(string name, long step, byte[] imageData, int width, int height, DateTime? wallTime = null)
    {
        var dataPoint = new ImageDataPoint
        {
            Step = step,
            WallTime = wallTime ?? DateTime.UtcNow,
            Data = imageData,
            Width = width,
            Height = height
        };

        var series = _images.GetOrAdd(name, _ => new List<ImageDataPoint>());
        lock (series)
        {
            series.Add(dataPoint);
        }

        CheckAutoSave();
    }

    /// <inheritdoc />
    public void LogText(string name, long step, string text, DateTime? wallTime = null)
    {
        var dataPoint = new TextDataPoint
        {
            Step = step,
            WallTime = wallTime ?? DateTime.UtcNow,
            Text = text
        };

        var series = _texts.GetOrAdd(name, _ => new List<TextDataPoint>());
        lock (series)
        {
            series.Add(dataPoint);
        }

        CheckAutoSave();
    }

    /// <inheritdoc />
    public void LogHyperparameters(Dictionary<string, object> hyperparams, Dictionary<string, double>? metrics = null)
    {
        var entry = new Dictionary<string, object>(hyperparams);
        if (metrics is not null)
        {
            entry["_metrics"] = metrics;
        }
        entry["_logged_at"] = DateTime.UtcNow;

        lock (_lock)
        {
            _hyperparameters.Add(entry);
        }

        CheckAutoSave();
    }

    /// <inheritdoc />
    public void LogConfusionMatrix(string name, long step, int[,] matrix, string[] labels, DateTime? wallTime = null)
    {
        var dataPoint = new ConfusionMatrixDataPoint
        {
            Step = step,
            WallTime = wallTime ?? DateTime.UtcNow,
            Matrix = matrix,
            Labels = labels
        };

        var series = _confusionMatrices.GetOrAdd(name, _ => new List<ConfusionMatrixDataPoint>());
        lock (series)
        {
            series.Add(dataPoint);
        }

        CheckAutoSave();
    }

    /// <inheritdoc />
    public void LogPRCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null)
    {
        var curve = CalculatePRCurve(predictions, labels);
        curve.Step = step;
        curve.WallTime = wallTime ?? DateTime.UtcNow;

        var series = _prCurves.GetOrAdd(name, _ => new List<CurveDataPoint>());
        lock (series)
        {
            series.Add(curve);
        }

        CheckAutoSave();
    }

    /// <inheritdoc />
    public void LogROCCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null)
    {
        var curve = CalculateROCCurve(predictions, labels);
        curve.Step = step;
        curve.WallTime = wallTime ?? DateTime.UtcNow;

        var series = _rocCurves.GetOrAdd(name, _ => new List<CurveDataPoint>());
        lock (series)
        {
            series.Add(curve);
        }

        CheckAutoSave();
    }

    /// <inheritdoc />
    public void LogModelGraph(string modelDescription)
    {
        _modelGraph = modelDescription;
    }

    /// <inheritdoc />
    public string GenerateReport(string? outputPath = null)
    {
        var path = outputPath ?? Path.Combine(LogDirectory, $"{Name}_report.html");
        var html = GenerateHtmlContent();
        File.WriteAllText(path, html, Encoding.UTF8);
        return path;
    }

    /// <inheritdoc />
    public void ExportTensorBoardFormat(string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);

        // Export scalars as JSON (TensorBoard can import via custom plugins)
        var scalarsPath = Path.Combine(outputDirectory, "scalars.json");
        var scalarsData = GetScalarData();
        File.WriteAllText(scalarsPath, JsonConvert.SerializeObject(scalarsData, Formatting.Indented));

        // Export histograms
        var histogramsPath = Path.Combine(outputDirectory, "histograms.json");
        var histogramsData = GetHistogramData();
        File.WriteAllText(histogramsPath, JsonConvert.SerializeObject(histogramsData, Formatting.Indented));

        // Export metadata
        var metadataPath = Path.Combine(outputDirectory, "metadata.json");
        var metadata = new
        {
            name = Name,
            exportedAt = DateTime.UtcNow,
            scalars = scalarsData.Keys.ToList(),
            histograms = histogramsData.Keys.ToList()
        };
        File.WriteAllText(metadataPath, JsonConvert.SerializeObject(metadata, Formatting.Indented));
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
        _images.Clear();
        _texts.Clear();
        _confusionMatrices.Clear();
        _prCurves.Clear();
        _rocCurves.Clear();
        _hyperparameters.Clear();
        _modelGraph = null;
        _logCount = 0;
    }

    /// <inheritdoc />
    public void Flush()
    {
        if (AutoSave)
        {
            GenerateReport();
        }
    }

    private void CheckAutoSave()
    {
        if (!AutoSave) return;

        // Use Interlocked for thread-safe increment and check
        int currentCount = System.Threading.Interlocked.Increment(ref _logCount);
        if (currentCount >= AutoSaveInterval)
        {
            // Use CompareExchange to ensure only one thread resets and generates report
            if (System.Threading.Interlocked.CompareExchange(ref _logCount, 0, currentCount) == currentCount)
            {
                GenerateReport();
            }
        }
    }

    private string GenerateHtmlContent()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang=\"en\">");
        sb.AppendLine("<head>");
        sb.AppendLine("    <meta charset=\"UTF-8\">");
        sb.AppendLine("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
        sb.AppendLine($"    <title>Training Dashboard - {EscapeHtml(Name)}</title>");
        sb.AppendLine("    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>");
        sb.AppendLine("    <script src=\"https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom\"></script>");
        sb.AppendLine(GetStyles());
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");

        // Header
        sb.AppendLine("<header>");
        sb.AppendLine($"    <h1>Training Dashboard: {EscapeHtml(Name)}</h1>");
        sb.AppendLine($"    <p class=\"subtitle\">Generated at {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC</p>");
        sb.AppendLine("</header>");

        // Navigation
        sb.AppendLine("<nav>");
        sb.AppendLine("    <a href=\"#scalars\" class=\"nav-link\">Scalars</a>");
        if (_histograms.Count > 0) sb.AppendLine("    <a href=\"#histograms\" class=\"nav-link\">Histograms</a>");
        if (_confusionMatrices.Count > 0) sb.AppendLine("    <a href=\"#confusion\" class=\"nav-link\">Confusion Matrix</a>");
        if (_prCurves.Count > 0 || _rocCurves.Count > 0) sb.AppendLine("    <a href=\"#curves\" class=\"nav-link\">ROC/PR Curves</a>");
        if (_hyperparameters.Count > 0) sb.AppendLine("    <a href=\"#hyperparams\" class=\"nav-link\">Hyperparameters</a>");
        if (!string.IsNullOrEmpty(_modelGraph)) sb.AppendLine("    <a href=\"#model\" class=\"nav-link\">Model</a>");
        sb.AppendLine("</nav>");

        sb.AppendLine("<main>");

        // Summary section
        sb.AppendLine("<section id=\"summary\" class=\"card\">");
        sb.AppendLine("    <h2>Training Summary</h2>");
        sb.AppendLine(GenerateSummaryHtml());
        sb.AppendLine("</section>");

        // Scalars section
        if (_scalars.Count > 0)
        {
            sb.AppendLine("<section id=\"scalars\" class=\"card\">");
            sb.AppendLine("    <h2>Scalar Metrics</h2>");
            sb.AppendLine(GenerateScalarsHtml());
            sb.AppendLine("</section>");
        }

        // Histograms section
        if (_histograms.Count > 0)
        {
            sb.AppendLine("<section id=\"histograms\" class=\"card\">");
            sb.AppendLine("    <h2>Histograms</h2>");
            sb.AppendLine(GenerateHistogramsHtml());
            sb.AppendLine("</section>");
        }

        // Confusion matrices section
        if (_confusionMatrices.Count > 0)
        {
            sb.AppendLine("<section id=\"confusion\" class=\"card\">");
            sb.AppendLine("    <h2>Confusion Matrices</h2>");
            sb.AppendLine(GenerateConfusionMatrixHtml());
            sb.AppendLine("</section>");
        }

        // ROC/PR Curves section
        if (_prCurves.Count > 0 || _rocCurves.Count > 0)
        {
            sb.AppendLine("<section id=\"curves\" class=\"card\">");
            sb.AppendLine("    <h2>ROC and PR Curves</h2>");
            sb.AppendLine(GenerateCurvesHtml());
            sb.AppendLine("</section>");
        }

        // Hyperparameters section
        if (_hyperparameters.Count > 0)
        {
            sb.AppendLine("<section id=\"hyperparams\" class=\"card\">");
            sb.AppendLine("    <h2>Hyperparameters</h2>");
            sb.AppendLine(GenerateHyperparamsHtml());
            sb.AppendLine("</section>");
        }

        // Model graph section
        var modelGraph = _modelGraph;
        if (!string.IsNullOrEmpty(modelGraph))
        {
            sb.AppendLine("<section id=\"model\" class=\"card\">");
            sb.AppendLine("    <h2>Model Architecture</h2>");
            sb.AppendLine($"    <pre class=\"model-graph\">{EscapeHtml(modelGraph)}</pre>");
            sb.AppendLine("</section>");
        }

        sb.AppendLine("</main>");

        sb.AppendLine("<footer>");
        sb.AppendLine("    <p>Generated by AiDotNet Training Infrastructure</p>");
        sb.AppendLine("</footer>");

        sb.AppendLine(GenerateScripts());
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        return sb.ToString();
    }

    private string GetStyles()
    {
        return @"
    <style>
        :root {
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }

        header {
            background: linear-gradient(135deg, var(--primary-dark), var(--primary));
            padding: 2rem;
            text-align: center;
        }

        header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        header .subtitle { color: rgba(255,255,255,0.8); }

        nav {
            background: var(--bg-card);
            padding: 1rem;
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-link {
            color: var(--text-muted);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s;
        }

        .nav-link:hover {
            background: var(--primary);
            color: white;
        }

        main {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            gap: 2rem;
        }

        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border);
        }

        .card h2 {
            color: var(--primary);
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .summary-item {
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }

        .summary-item .label { color: var(--text-muted); font-size: 0.875rem; }
        .summary-item .value { font-size: 1.5rem; font-weight: bold; color: var(--primary); }

        .chart-container {
            position: relative;
            height: 300px;
            margin: 1rem 0;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1rem;
        }

        .chart-card {
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
        }

        .chart-card h3 {
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th { color: var(--text-muted); font-weight: 500; }

        .matrix-container {
            overflow-x: auto;
        }

        .confusion-matrix {
            display: inline-block;
            margin: 1rem 0;
        }

        .matrix-cell {
            width: 50px;
            height: 50px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.875rem;
            border: 1px solid var(--border);
        }

        .model-graph {
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.875rem;
            white-space: pre-wrap;
        }

        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
        }

        @media (max-width: 768px) {
            main { padding: 1rem; }
            .charts-grid { grid-template-columns: 1fr; }
        }
    </style>";
    }

    private string GenerateSummaryHtml()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<div class=\"summary-grid\">");

        // Total metrics
        sb.AppendLine("<div class=\"summary-item\">");
        sb.AppendLine("    <div class=\"label\">Total Metrics</div>");
        sb.AppendLine($"    <div class=\"value\">{_scalars.Count}</div>");
        sb.AppendLine("</div>");

        // Total data points
        var totalPoints = _scalars.Values.Sum(s => s.Count);
        sb.AppendLine("<div class=\"summary-item\">");
        sb.AppendLine("    <div class=\"label\">Data Points</div>");
        sb.AppendLine($"    <div class=\"value\">{totalPoints:N0}</div>");
        sb.AppendLine("</div>");

        // Latest loss (if exists)
        var lossKey = _scalars.Keys.FirstOrDefault(k => k.Contains("loss", StringComparison.OrdinalIgnoreCase));
        if (lossKey is not null && _scalars[lossKey].Count > 0)
        {
            var latestLoss = _scalars[lossKey].Last().Value;
            sb.AppendLine("<div class=\"summary-item\">");
            sb.AppendLine("    <div class=\"label\">Latest Loss</div>");
            sb.AppendLine($"    <div class=\"value\">{latestLoss:F4}</div>");
            sb.AppendLine("</div>");
        }

        // Latest accuracy (if exists)
        var accKey = _scalars.Keys.FirstOrDefault(k => k.Contains("accuracy", StringComparison.OrdinalIgnoreCase) || k.Contains("acc", StringComparison.OrdinalIgnoreCase));
        if (accKey is not null && _scalars[accKey].Count > 0)
        {
            var latestAcc = _scalars[accKey].Last().Value;
            sb.AppendLine("<div class=\"summary-item\">");
            sb.AppendLine("    <div class=\"label\">Latest Accuracy</div>");
            sb.AppendLine($"    <div class=\"value\">{latestAcc:P1}</div>");
            sb.AppendLine("</div>");
        }

        // Training duration
        if (_scalars.Values.Any(s => s.Count > 0))
        {
            var allTimes = _scalars.Values.SelectMany(s => s.Select(p => p.WallTime)).ToList();
            if (allTimes.Count > 1)
            {
                var duration = allTimes.Max() - allTimes.Min();
                sb.AppendLine("<div class=\"summary-item\">");
                sb.AppendLine("    <div class=\"label\">Duration</div>");
                sb.AppendLine($"    <div class=\"value\">{duration:hh\\:mm\\:ss}</div>");
                sb.AppendLine("</div>");
            }
        }

        sb.AppendLine("</div>");
        return sb.ToString();
    }

    private string GenerateScalarsHtml()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<div class=\"charts-grid\">");

        int chartIndex = 0;
        foreach (var kvp in _scalars.OrderBy(k => k.Key))
        {
            var chartId = $"scalar_chart_{chartIndex++}";
            sb.AppendLine($"<div class=\"chart-card\">");
            sb.AppendLine($"    <h3>{EscapeHtml(kvp.Key)}</h3>");
            sb.AppendLine($"    <div class=\"chart-container\"><canvas id=\"{chartId}\"></canvas></div>");
            sb.AppendLine("</div>");
        }

        sb.AppendLine("</div>");
        return sb.ToString();
    }

    private string GenerateHistogramsHtml()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<div class=\"charts-grid\">");

        int chartIndex = 0;
        foreach (var kvp in _histograms.OrderBy(k => k.Key))
        {
            if (kvp.Value.Count == 0) continue;

            var latest = kvp.Value.Last();
            var chartId = $"hist_chart_{chartIndex++}";

            sb.AppendLine($"<div class=\"chart-card\">");
            sb.AppendLine($"    <h3>{EscapeHtml(kvp.Key)} (Step {latest.Step})</h3>");
            sb.AppendLine($"    <div class=\"chart-container\"><canvas id=\"{chartId}\"></canvas></div>");
            sb.AppendLine($"    <p style=\"color: var(--text-muted); font-size: 0.75rem;\">Mean: {latest.Mean:F4}, StdDev: {latest.StdDev:F4}</p>");
            sb.AppendLine("</div>");
        }

        sb.AppendLine("</div>");
        return sb.ToString();
    }

    private string GenerateConfusionMatrixHtml()
    {
        var sb = new StringBuilder();

        foreach (var kvp in _confusionMatrices)
        {
            if (kvp.Value.Count == 0) continue;

            var latest = kvp.Value.Last();
            var rows = latest.Matrix.GetLength(0);
            var cols = latest.Matrix.GetLength(1);
            var maxVal = 0;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    maxVal = Math.Max(maxVal, latest.Matrix[i, j]);

            sb.AppendLine($"<h3>{EscapeHtml(kvp.Key)} (Step {latest.Step})</h3>");
            sb.AppendLine("<div class=\"matrix-container\">");
            sb.AppendLine("<table style=\"display: inline-block;\">");

            // Header row
            sb.AppendLine("<tr><th></th>");
            foreach (var label in latest.Labels)
            {
                sb.AppendLine($"<th style=\"text-align: center;\">{EscapeHtml(label)}</th>");
            }
            sb.AppendLine("</tr>");

            // Data rows
            for (int i = 0; i < rows; i++)
            {
                sb.AppendLine($"<tr><th>{EscapeHtml(latest.Labels[i])}</th>");
                for (int j = 0; j < cols; j++)
                {
                    var val = latest.Matrix[i, j];
                    var intensity = maxVal > 0 ? (double)val / maxVal : 0;
                    var bgColor = i == j
                        ? $"rgba(34, 197, 94, {0.2 + intensity * 0.6})"
                        : $"rgba(239, 68, 68, {intensity * 0.5})";
                    sb.AppendLine($"<td style=\"text-align: center; background: {bgColor};\">{val}</td>");
                }
                sb.AppendLine("</tr>");
            }

            sb.AppendLine("</table>");
            sb.AppendLine("</div>");
        }

        return sb.ToString();
    }

    private string GenerateCurvesHtml()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<div class=\"charts-grid\">");

        int chartIndex = 0;

        foreach (var kvp in _rocCurves)
        {
            if (kvp.Value.Count == 0) continue;
            var latest = kvp.Value.Last();
            var chartId = $"roc_chart_{chartIndex++}";

            sb.AppendLine($"<div class=\"chart-card\">");
            sb.AppendLine($"    <h3>ROC Curve: {EscapeHtml(kvp.Key)} (AUC: {latest.AUC:F3})</h3>");
            sb.AppendLine($"    <div class=\"chart-container\"><canvas id=\"{chartId}\"></canvas></div>");
            sb.AppendLine("</div>");
        }

        foreach (var kvp in _prCurves)
        {
            if (kvp.Value.Count == 0) continue;
            var latest = kvp.Value.Last();
            var chartId = $"pr_chart_{chartIndex++}";

            sb.AppendLine($"<div class=\"chart-card\">");
            sb.AppendLine($"    <h3>PR Curve: {EscapeHtml(kvp.Key)} (AP: {latest.AUC:F3})</h3>");
            sb.AppendLine($"    <div class=\"chart-container\"><canvas id=\"{chartId}\"></canvas></div>");
            sb.AppendLine("</div>");
        }

        sb.AppendLine("</div>");
        return sb.ToString();
    }

    private string GenerateHyperparamsHtml()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<table>");
        sb.AppendLine("<tr><th>Parameter</th><th>Value</th></tr>");

        foreach (var entry in _hyperparameters)
        {
            foreach (var kvp in entry.Where(k => !k.Key.StartsWith("_")))
            {
                sb.AppendLine($"<tr><td>{EscapeHtml(kvp.Key)}</td><td>{EscapeHtml(kvp.Value?.ToString() ?? "null")}</td></tr>");
            }
        }

        sb.AppendLine("</table>");
        return sb.ToString();
    }

    private string GenerateScripts()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<script>");

        // Generate scalar charts
        int chartIndex = 0;
        foreach (var kvp in _scalars.OrderBy(k => k.Key))
        {
            var chartId = $"scalar_chart_{chartIndex++}";
            var data = kvp.Value.OrderBy(p => p.Step).ToList();
            var labels = JsonConvert.SerializeObject(data.Select(p => p.Step).ToArray());
            var values = JsonConvert.SerializeObject(data.Select(p => p.Value).ToArray());

            sb.AppendLine($@"
new Chart(document.getElementById('{chartId}'), {{
    type: 'line',
    data: {{
        labels: {labels},
        datasets: [{{
            label: '{EscapeJs(kvp.Key)}',
            data: {values},
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1,
            pointRadius: 0,
            pointHitRadius: 10
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            x: {{ title: {{ display: true, text: 'Step', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            y: {{ title: {{ display: true, text: 'Value', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }},
        plugins: {{
            legend: {{ display: false }},
            zoom: {{ zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: 'xy' }}, pan: {{ enabled: true, mode: 'xy' }} }}
        }}
    }}
}});");
        }

        // Generate histogram charts
        chartIndex = 0;
        foreach (var kvp in _histograms.OrderBy(k => k.Key))
        {
            if (kvp.Value.Count == 0) continue;
            var latest = kvp.Value.Last();
            var chartId = $"hist_chart_{chartIndex++}";
            var labels = JsonConvert.SerializeObject(latest.BucketLimits.Select(l => l.ToString("F3", CultureInfo.InvariantCulture)).ToArray());
            var values = JsonConvert.SerializeObject(latest.BucketCounts);

            sb.AppendLine($@"
new Chart(document.getElementById('{chartId}'), {{
    type: 'bar',
    data: {{
        labels: {labels},
        datasets: [{{
            label: 'Count',
            data: {values},
            backgroundColor: 'rgba(59, 130, 246, 0.6)',
            borderColor: '#3b82f6',
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            x: {{ title: {{ display: true, text: 'Value', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            y: {{ title: {{ display: true, text: 'Count', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }},
        plugins: {{ legend: {{ display: false }} }}
    }}
}});");
        }

        // Generate ROC curves
        chartIndex = 0;
        foreach (var kvp in _rocCurves)
        {
            if (kvp.Value.Count == 0) continue;
            var latest = kvp.Value.Last();
            var chartId = $"roc_chart_{chartIndex++}";
            var xValues = JsonConvert.SerializeObject(latest.XValues);
            var yValues = JsonConvert.SerializeObject(latest.YValues);

            sb.AppendLine($@"
new Chart(document.getElementById('{chartId}'), {{
    type: 'line',
    data: {{
        datasets: [
            {{
                label: 'ROC Curve',
                data: {xValues}.map((x, i) => ({{ x: x, y: {yValues}[i] }})),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0
            }},
            {{
                label: 'Random',
                data: [{{x: 0, y: 0}}, {{x: 1, y: 1}}],
                borderColor: '#94a3b8',
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            x: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'False Positive Rate', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            y: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'True Positive Rate', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }}
    }}
}});");
        }

        // Generate PR curves
        foreach (var kvp in _prCurves)
        {
            if (kvp.Value.Count == 0) continue;
            var latest = kvp.Value.Last();
            var chartId = $"pr_chart_{chartIndex++}";
            var xValues = JsonConvert.SerializeObject(latest.XValues);
            var yValues = JsonConvert.SerializeObject(latest.YValues);

            sb.AppendLine($@"
new Chart(document.getElementById('{chartId}'), {{
    type: 'line',
    data: {{
        datasets: [{{
            label: 'PR Curve',
            data: {xValues}.map((x, i) => ({{ x: x, y: {yValues}[i] }})),
            borderColor: '#22c55e',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            fill: true,
            tension: 0
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            x: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'Recall', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            y: {{ type: 'linear', min: 0, max: 1, title: {{ display: true, text: 'Precision', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }},
        plugins: {{ legend: {{ display: false }} }}
    }}
}});");
        }

        sb.AppendLine("</script>");
        return sb.ToString();
    }

    private static CurveDataPoint CalculatePRCurve(double[] predictions, int[] labels)
    {
        var sorted = predictions.Zip(labels, (p, l) => (pred: p, label: l))
            .OrderByDescending(x => x.pred)
            .ToList();

        int totalPositives = labels.Sum();
        int tp = 0, fp = 0;

        var precisions = new List<double>();
        var recalls = new List<double>();
        var thresholds = new List<double>();

        for (int i = 0; i < sorted.Count; i++)
        {
            if (sorted[i].label == 1) tp++;
            else fp++;

            var precision = (double)tp / (tp + fp);
            var recall = totalPositives > 0 ? (double)tp / totalPositives : 0;

            precisions.Add(precision);
            recalls.Add(recall);
            thresholds.Add(sorted[i].pred);
        }

        // Calculate Average Precision (AUC for PR curve)
        double ap = 0;
        for (int i = 1; i < recalls.Count; i++)
        {
            ap += (recalls[i] - recalls[i - 1]) * precisions[i];
        }

        return new CurveDataPoint
        {
            XValues = recalls.ToArray(),
            YValues = precisions.ToArray(),
            Thresholds = thresholds.ToArray(),
            AUC = ap
        };
    }

    private static CurveDataPoint CalculateROCCurve(double[] predictions, int[] labels)
    {
        var sorted = predictions.Zip(labels, (p, l) => (pred: p, label: l))
            .OrderByDescending(x => x.pred)
            .ToList();

        int totalPositives = labels.Sum();
        int totalNegatives = labels.Length - totalPositives;
        int tp = 0, fp = 0;

        var tprs = new List<double> { 0 };
        var fprs = new List<double> { 0 };
        var thresholds = new List<double>();

        for (int i = 0; i < sorted.Count; i++)
        {
            if (sorted[i].label == 1) tp++;
            else fp++;

            var tpr = totalPositives > 0 ? (double)tp / totalPositives : 0;
            var fpr = totalNegatives > 0 ? (double)fp / totalNegatives : 0;

            tprs.Add(tpr);
            fprs.Add(fpr);
            thresholds.Add(sorted[i].pred);
        }

        // Calculate AUC using trapezoidal rule
        double auc = 0;
        for (int i = 1; i < fprs.Count; i++)
        {
            auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2;
        }

        return new CurveDataPoint
        {
            XValues = fprs.ToArray(),
            YValues = tprs.ToArray(),
            Thresholds = thresholds.ToArray(),
            AUC = auc
        };
    }

    private static string EscapeHtml(string text)
    {
        return System.Net.WebUtility.HtmlEncode(text);
    }

    private static string EscapeJs(string text)
    {
        return text.Replace("\\", "\\\\").Replace("'", "\\'").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "\\r");
    }

    /// <summary>
    /// Disposes the dashboard.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        Stop();
        Flush();
    }
}
