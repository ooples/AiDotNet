using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using SysConsole = System.Console;

namespace AiDotNet.Dashboard.Interpretability;

/// <summary>
/// Unified dashboard for displaying and exporting interpretability explanations.
/// Supports multiple explanation types: feature attribution, heatmaps, counterfactuals.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This dashboard provides a one-stop interface for visualizing
/// model explanations. Whether you're using SHAP, LIME, GradCAM, or other methods,
/// this dashboard can display and export the results.
///
/// <b>Key features:</b>
/// - Console visualization (ASCII charts)
/// - HTML export (interactive web pages)
/// - Multi-format export (JSON, CSV, Markdown)
/// - Session-based explanation tracking
/// - Comparison between explanations
/// </para>
/// </remarks>
public class InterpretabilityDashboard : IDisposable
{
    private readonly string _title;
    private readonly AttributionVisualizer _visualizer;
    private readonly ExplanationExporter? _exporter;
    private readonly Dictionary<string, ExplanationSession> _sessions;
    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Gets the output directory for exports, or null if no directory was specified.
    /// </summary>
    public string? OutputDirectory { get; }

    /// <summary>
    /// Initializes a new interpretability dashboard.
    /// </summary>
    /// <param name="title">Dashboard title.</param>
    /// <param name="outputDirectory">Optional directory for exports.</param>
    /// <param name="visualizerWidth">Width of console visualizations.</param>
    /// <param name="useColor">Whether to use console colors.</param>
    public InterpretabilityDashboard(
        string title = "Model Interpretability",
        string? outputDirectory = null,
        int visualizerWidth = 100,
        bool useColor = true)
    {
        _title = title;
        _visualizer = new AttributionVisualizer(visualizerWidth, 20, useColor);
        _sessions = new Dictionary<string, ExplanationSession>();
        OutputDirectory = outputDirectory;

        if (!string.IsNullOrEmpty(outputDirectory))
        {
            _exporter = new ExplanationExporter(outputDirectory);
        }
    }

    /// <summary>
    /// Starts a new explanation session.
    /// </summary>
    /// <param name="sessionName">Name for the session.</param>
    /// <param name="methodName">Explanation method (SHAP, LIME, etc.).</param>
    /// <param name="modelName">Optional model name.</param>
    /// <returns>Session ID.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A session groups related explanations together. For example,
    /// you might create a session for "Customer Churn Predictions" and add multiple
    /// individual explanations to it.
    /// </para>
    /// </remarks>
    public string StartSession(string sessionName, string methodName, string? modelName = null)
    {
        lock (_lock)
        {
            string sessionId = Guid.NewGuid().ToString("N")[..8];

            _sessions[sessionId] = new ExplanationSession
            {
                Id = sessionId,
                Name = sessionName,
                MethodName = methodName,
                ModelName = modelName,
                StartTime = DateTime.UtcNow,
                Explanations = new List<Explanation>()
            };

            return sessionId;
        }
    }

    /// <summary>
    /// Logs a feature attribution explanation.
    /// </summary>
    /// <param name="sessionId">Session ID from StartSession.</param>
    /// <param name="instanceId">Unique identifier for the explained instance.</param>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="prediction">Model prediction for the instance.</param>
    /// <param name="baseValue">Optional base value (expected value).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this after computing SHAP, LIME, or other attributions
    /// to record them in the dashboard. The instanceId helps you track which data
    /// point each explanation refers to.
    /// </para>
    /// </remarks>
    public void LogAttribution(
        string sessionId,
        string instanceId,
        string[] featureNames,
        double[] attributions,
        double? prediction = null,
        double? baseValue = null)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            session.Explanations.Add(new Explanation
            {
                InstanceId = instanceId,
                Type = ExplanationType.FeatureAttribution,
                FeatureNames = featureNames,
                Attributions = attributions,
                Prediction = prediction,
                BaseValue = baseValue,
                Timestamp = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Logs a spatial/heatmap attribution (like GradCAM).
    /// </summary>
    /// <param name="sessionId">Session ID.</param>
    /// <param name="instanceId">Instance identifier.</param>
    /// <param name="heatmap">2D attribution heatmap.</param>
    /// <param name="prediction">Model prediction.</param>
    /// <param name="targetClass">Target class being explained.</param>
    public void LogHeatmap(
        string sessionId,
        string instanceId,
        double[,] heatmap,
        double? prediction = null,
        int? targetClass = null)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            session.Explanations.Add(new Explanation
            {
                InstanceId = instanceId,
                Type = ExplanationType.SpatialHeatmap,
                Heatmap = heatmap,
                Prediction = prediction,
                TargetClass = targetClass,
                Timestamp = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Logs a counterfactual explanation.
    /// </summary>
    /// <param name="sessionId">Session ID.</param>
    /// <param name="instanceId">Instance identifier.</param>
    /// <param name="featureNames">Feature names.</param>
    /// <param name="originalValues">Original feature values.</param>
    /// <param name="counterfactualValues">Changed feature values.</param>
    /// <param name="originalPrediction">Original prediction.</param>
    /// <param name="counterfactualPrediction">Prediction after changes.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Counterfactuals answer "What would need to change to get
    /// a different prediction?" This method records both the original and changed values.
    /// </para>
    /// </remarks>
    public void LogCounterfactual(
        string sessionId,
        string instanceId,
        string[] featureNames,
        double[] originalValues,
        double[] counterfactualValues,
        double originalPrediction,
        double counterfactualPrediction)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            session.Explanations.Add(new Explanation
            {
                InstanceId = instanceId,
                Type = ExplanationType.Counterfactual,
                FeatureNames = featureNames,
                OriginalValues = originalValues,
                CounterfactualValues = counterfactualValues,
                Prediction = counterfactualPrediction,
                OriginalPrediction = originalPrediction,
                Timestamp = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Displays a single attribution explanation to the console.
    /// </summary>
    /// <param name="sessionId">Session ID.</param>
    /// <param name="instanceId">Instance to display.</param>
    /// <param name="topK">Show only top K features.</param>
    public void ShowAttribution(string sessionId, string instanceId, int topK = 15)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            var explanation = session.Explanations
                .FirstOrDefault(e => e.InstanceId == instanceId && e.Type == ExplanationType.FeatureAttribution);

            if (explanation == null)
            {
                SysConsole.WriteLine($"No attribution found for instance {instanceId}");
                return;
            }

            string title = $"{session.MethodName} Attribution - Instance: {instanceId}";
            if (explanation.Prediction.HasValue)
            {
                title += $" (Prediction: {explanation.Prediction.Value:F4})";
            }

            if (explanation.FeatureNames != null && explanation.Attributions != null)
            {
                _visualizer.RenderBarChart(
                    explanation.FeatureNames,
                    explanation.Attributions,
                    title,
                    topK);

                if (explanation.BaseValue.HasValue)
                {
                    _visualizer.RenderWaterfallChart(
                        explanation.FeatureNames,
                        explanation.Attributions,
                        explanation.BaseValue.Value,
                        "Contribution Waterfall");
                }
            }
        }
    }

    /// <summary>
    /// Displays a heatmap explanation to the console.
    /// </summary>
    /// <param name="sessionId">Session ID.</param>
    /// <param name="instanceId">Instance to display.</param>
    public void ShowHeatmap(string sessionId, string instanceId)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            var explanation = session.Explanations
                .FirstOrDefault(e => e.InstanceId == instanceId && e.Type == ExplanationType.SpatialHeatmap);

            if (explanation?.Heatmap == null)
            {
                SysConsole.WriteLine($"No heatmap found for instance {instanceId}");
                return;
            }

            string title = $"{session.MethodName} Heatmap - Instance: {instanceId}";
            if (explanation.TargetClass.HasValue)
            {
                title += $" (Class: {explanation.TargetClass.Value})";
            }

            _visualizer.RenderHeatmap(explanation.Heatmap, title);
        }
    }

    /// <summary>
    /// Displays a counterfactual explanation to the console.
    /// </summary>
    /// <param name="sessionId">Session ID.</param>
    /// <param name="instanceId">Instance to display.</param>
    public void ShowCounterfactual(string sessionId, string instanceId)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            var explanation = session.Explanations
                .FirstOrDefault(e => e.InstanceId == instanceId && e.Type == ExplanationType.Counterfactual);

            if (explanation == null)
            {
                SysConsole.WriteLine($"No counterfactual found for instance {instanceId}");
                return;
            }

            SysConsole.WriteLine();
            SysConsole.WriteLine($"Counterfactual Explanation - Instance: {instanceId}");
            SysConsole.WriteLine(new string('=', 60));
            SysConsole.WriteLine();

            if (explanation.OriginalPrediction.HasValue && explanation.Prediction.HasValue)
            {
                SysConsole.WriteLine($"Original Prediction:      {explanation.OriginalPrediction.Value:F4}");
                SysConsole.WriteLine($"Counterfactual Prediction: {explanation.Prediction.Value:F4}");
                SysConsole.WriteLine();
            }

            if (explanation.FeatureNames != null && explanation.OriginalValues != null && explanation.CounterfactualValues != null)
            {
                SysConsole.WriteLine("Changes Required:");
                SysConsole.WriteLine(new string('-', 60));
                SysConsole.WriteLine($"{"Feature",-25} {"Original",12} {"Changed",12} {"Delta",12}");
                SysConsole.WriteLine(new string('-', 60));

                for (int i = 0; i < explanation.FeatureNames.Length; i++)
                {
                    double delta = explanation.CounterfactualValues[i] - explanation.OriginalValues[i];
                    if (Math.Abs(delta) > 1e-6)
                    {
                        SysConsole.ForegroundColor = delta > 0 ? System.ConsoleColor.Green : System.ConsoleColor.Red;
                        SysConsole.WriteLine($"{explanation.FeatureNames[i],-25} {explanation.OriginalValues[i],12:F4} {explanation.CounterfactualValues[i],12:F4} {(delta >= 0 ? "+" : "")}{delta,11:F4}");
                        SysConsole.ResetColor();
                    }
                }
            }

            SysConsole.WriteLine();
        }
    }

    /// <summary>
    /// Shows a summary of all explanations in a session.
    /// </summary>
    /// <param name="sessionId">Session ID.</param>
    public void ShowSessionSummary(string sessionId)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            SysConsole.WriteLine();
            SysConsole.WriteLine($"Session: {session.Name}");
            SysConsole.WriteLine(new string('=', 60));
            SysConsole.WriteLine($"Method: {session.MethodName}");
            if (!string.IsNullOrEmpty(session.ModelName))
            {
                SysConsole.WriteLine($"Model: {session.ModelName}");
            }
            SysConsole.WriteLine($"Started: {session.StartTime:yyyy-MM-dd HH:mm:ss}");
            SysConsole.WriteLine($"Total Explanations: {session.Explanations.Count}");
            SysConsole.WriteLine();

            // Group by type
            var byType = session.Explanations.GroupBy(e => e.Type);
            foreach (var group in byType)
            {
                SysConsole.WriteLine($"  {group.Key}: {group.Count()} explanations");
            }

            // If feature attributions exist, show global summary
            var attributions = session.Explanations
                .Where(e => e.Type == ExplanationType.FeatureAttribution && e.FeatureNames != null)
                .ToList();

            if (attributions.Count > 0)
            {
                SysConsole.WriteLine();
                SysConsole.WriteLine("Global Feature Importance (Mean |Attribution|):");
                SysConsole.WriteLine(new string('-', 50));

                var featureNames = attributions[0].FeatureNames!;
                var meanAbsAttr = new double[featureNames.Length];

                foreach (var exp in attributions)
                {
                    if (exp.Attributions != null)
                    {
                        for (int i = 0; i < featureNames.Length && i < exp.Attributions.Length; i++)
                        {
                            meanAbsAttr[i] += Math.Abs(exp.Attributions[i]);
                        }
                    }
                }

                for (int i = 0; i < meanAbsAttr.Length; i++)
                {
                    meanAbsAttr[i] /= attributions.Count;
                }

                _visualizer.RenderBarChart(featureNames, meanAbsAttr, null, 10);
            }
        }
    }

    /// <summary>
    /// Exports a session to various formats.
    /// </summary>
    /// <param name="sessionId">Session ID.</param>
    /// <param name="format">Export format (json, csv, html, markdown, all).</param>
    /// <returns>Dictionary of format to file path.</returns>
    public Dictionary<string, string> ExportSession(string sessionId, string format = "all")
    {
        if (_exporter == null)
        {
            throw new InvalidOperationException("No output directory specified. Create dashboard with outputDirectory parameter.");
        }

        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            var results = new Dictionary<string, string>();
            string baseFilename = $"{session.Name.Replace(" ", "_")}_{sessionId}";

            var metadata = new Dictionary<string, object>
            {
                ["session_id"] = sessionId,
                ["session_name"] = session.Name,
                ["method"] = session.MethodName,
                ["model"] = session.ModelName ?? "Unknown",
                ["start_time"] = session.StartTime.ToString("o"),
                ["explanation_count"] = session.Explanations.Count
            };

            // Get aggregated feature attributions
            var attributions = session.Explanations
                .Where(e => e.Type == ExplanationType.FeatureAttribution && e.FeatureNames != null && e.Attributions != null)
                .ToList();

            if (attributions.Count > 0)
            {
                var featureNames = attributions[0].FeatureNames!;
                var meanAttr = new double[featureNames.Length];

                foreach (var exp in attributions)
                {
                    for (int i = 0; i < featureNames.Length && i < exp.Attributions!.Length; i++)
                    {
                        meanAttr[i] += exp.Attributions[i];
                    }
                }

                for (int i = 0; i < meanAttr.Length; i++)
                {
                    meanAttr[i] /= attributions.Count;
                }

                if (format == "all" || format == "json")
                {
                    results["json"] = _exporter.ExportToJson(featureNames, meanAttr, metadata, baseFilename);
                }

                if (format == "all" || format == "csv")
                {
                    results["csv"] = _exporter.ExportToCsv(featureNames, meanAttr, baseFilename);
                }

                if (format == "all" || format == "html")
                {
                    results["html"] = _exporter.ExportToHtml(featureNames, meanAttr, $"{session.Name} - {session.MethodName}", metadata, baseFilename);
                }

                if (format == "all" || format == "markdown")
                {
                    results["markdown"] = _exporter.ExportToMarkdown(featureNames, meanAttr, $"{session.Name} - {session.MethodName}", metadata, baseFilename);
                }

                // Export individual instances
                if (attributions.Count > 1)
                {
                    var batchData = attributions
                        .Select(e => (e.InstanceId, e.Attributions!))
                        .ToList();

                    results["batch_json"] = _exporter.ExportBatchToJson(batchData, featureNames, metadata, baseFilename + "_batch");
                }
            }

            // Export heatmaps
            var heatmaps = session.Explanations
                .Where(e => e.Type == ExplanationType.SpatialHeatmap && e.Heatmap != null)
                .ToList();

            foreach (var heatmap in heatmaps)
            {
                string heatmapFilename = $"{baseFilename}_heatmap_{heatmap.InstanceId}";
                results[$"heatmap_{heatmap.InstanceId}"] = _exporter.ExportHeatmapToHtml(
                    heatmap.Heatmap!,
                    $"Heatmap - {heatmap.InstanceId}",
                    metadata,
                    heatmapFilename);
            }

            return results;
        }
    }

    /// <summary>
    /// Gets all session IDs.
    /// </summary>
    public IEnumerable<string> GetSessionIds()
    {
        lock (_lock)
        {
            return _sessions.Keys.ToList();
        }
    }

    /// <summary>
    /// Gets session information.
    /// </summary>
    public (string Name, string Method, int ExplanationCount) GetSessionInfo(string sessionId)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
            {
                throw new ArgumentException($"Session {sessionId} not found.");
            }

            return (session.Name, session.MethodName, session.Explanations.Count);
        }
    }

    /// <summary>
    /// Clears all sessions.
    /// </summary>
    public void ClearAllSessions()
    {
        lock (_lock)
        {
            _sessions.Clear();
        }
    }

    /// <summary>
    /// Disposes the dashboard resources.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }

    #region Private Types

    private class ExplanationSession
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string MethodName { get; set; } = string.Empty;
        public string? ModelName { get; set; }
        public DateTime StartTime { get; set; }
        public List<Explanation> Explanations { get; set; } = new();
    }

    private class Explanation
    {
        public string InstanceId { get; set; } = string.Empty;
        public ExplanationType Type { get; set; }
        public DateTime Timestamp { get; set; }

        // Feature attribution
        public string[]? FeatureNames { get; set; }
        public double[]? Attributions { get; set; }
        public double? Prediction { get; set; }
        public double? BaseValue { get; set; }

        // Heatmap
        public double[,]? Heatmap { get; set; }
        public int? TargetClass { get; set; }

        // Counterfactual
        public double[]? OriginalValues { get; set; }
        public double[]? CounterfactualValues { get; set; }
        public double? OriginalPrediction { get; set; }
    }

    private enum ExplanationType
    {
        FeatureAttribution,
        SpatialHeatmap,
        Counterfactual,
        NeuronAttribution,
        LayerAttribution
    }

    #endregion
}
