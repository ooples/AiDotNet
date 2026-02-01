using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Dashboard.Interpretability;

/// <summary>
/// Exports interpretability explanations to various formats (JSON, CSV, HTML).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> After computing explanations with SHAP, LIME, or other methods,
/// you often need to save or share them. This class lets you export explanations to:
/// - JSON (for programmatic access)
/// - CSV (for spreadsheet analysis)
/// - HTML (for interactive visualization)
/// - Markdown (for documentation)
/// </para>
/// </remarks>
public class ExplanationExporter
{
    private readonly string _outputDirectory;
    private readonly AttributionVisualizer _visualizer;

    /// <summary>
    /// Initializes a new explanation exporter.
    /// </summary>
    /// <param name="outputDirectory">Directory to save exported files.</param>
    public ExplanationExporter(string outputDirectory)
    {
        _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));

        // Create directory if it doesn't exist
        if (!Directory.Exists(_outputDirectory))
        {
            Directory.CreateDirectory(_outputDirectory);
        }

        _visualizer = new AttributionVisualizer();
    }

    /// <summary>
    /// Exports feature attributions to JSON format.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="metadata">Optional metadata (model name, method, etc.).</param>
    /// <param name="filename">Output filename (without extension).</param>
    /// <returns>Path to the exported file.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> JSON format is good for loading explanations back into code
    /// or sending them to a web application.
    /// </para>
    /// </remarks>
    public string ExportToJson(
        string[] featureNames,
        double[] attributions,
        Dictionary<string, object>? metadata = null,
        string filename = "explanation")
    {
        var exportData = new JObject
        {
            ["timestamp"] = DateTime.UtcNow.ToString("o"),
            ["format_version"] = "1.0"
        };

        // Add metadata
        if (metadata != null)
        {
            var metaObj = new JObject();
            foreach (var kvp in metadata)
            {
                metaObj[kvp.Key] = JToken.FromObject(kvp.Value);
            }
            exportData["metadata"] = metaObj;
        }

        // Add summary statistics
        var stats = new JObject
        {
            ["total_features"] = featureNames.Length,
            ["sum_attributions"] = attributions.Sum(),
            ["mean_attribution"] = attributions.Average(),
            ["max_attribution"] = attributions.Max(),
            ["min_attribution"] = attributions.Min(),
            ["std_attribution"] = CalculateStdDev(attributions)
        };
        exportData["statistics"] = stats;

        // Add feature attributions (sorted by absolute value)
        var sortedFeatures = featureNames
            .Zip(attributions, (name, attr) => new { Name = name, Attribution = attr })
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .ToList();

        var featuresArray = new JArray();
        int rank = 1;
        foreach (var feature in sortedFeatures)
        {
            featuresArray.Add(new JObject
            {
                ["rank"] = rank++,
                ["feature"] = feature.Name,
                ["attribution"] = feature.Attribution,
                ["abs_attribution"] = Math.Abs(feature.Attribution),
                ["direction"] = feature.Attribution >= 0 ? "positive" : "negative"
            });
        }
        exportData["features"] = featuresArray;

        // Write to file
        string filePath = GetSafeFilePath(filename, ".json");
        File.WriteAllText(filePath, exportData.ToString(Formatting.Indented));

        return filePath;
    }

    /// <summary>
    /// Exports feature attributions to CSV format.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="filename">Output filename (without extension).</param>
    /// <param name="includeHeader">Whether to include column headers.</param>
    /// <returns>Path to the exported file.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CSV format is great for opening in Excel or Google Sheets
    /// for further analysis.
    /// </para>
    /// </remarks>
    public string ExportToCsv(
        string[] featureNames,
        double[] attributions,
        string filename = "explanation",
        bool includeHeader = true)
    {
        var sb = new StringBuilder();

        if (includeHeader)
        {
            sb.AppendLine("rank,feature,attribution,abs_attribution,direction");
        }

        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .ToList();

        int rank = 1;
        foreach (var (name, attr) in sorted)
        {
            string escapedName = name.Contains(',') || name.Contains('"')
                ? $"\"{name.Replace("\"", "\"\"")}\""
                : name;

            sb.AppendLine($"{rank++},{escapedName},{attr.ToString(CultureInfo.InvariantCulture)},{Math.Abs(attr).ToString(CultureInfo.InvariantCulture)},{(attr >= 0 ? "positive" : "negative")}");
        }

        string filePath = GetSafeFilePath(filename, ".csv");
        File.WriteAllText(filePath, sb.ToString());

        return filePath;
    }

    /// <summary>
    /// Exports feature attributions to an interactive HTML page.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="title">Page title.</param>
    /// <param name="metadata">Optional metadata to display.</param>
    /// <param name="filename">Output filename (without extension).</param>
    /// <returns>Path to the exported file.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> HTML export creates an interactive visualization you can
    /// open in any web browser. Great for sharing with non-technical stakeholders.
    /// </para>
    /// </remarks>
    public string ExportToHtml(
        string[] featureNames,
        double[] attributions,
        string title = "Model Explanation",
        Dictionary<string, object>? metadata = null,
        string filename = "explanation")
    {
        string html = _visualizer.GenerateHtmlPage(featureNames, attributions, title);

        // Inject metadata if provided
        if (metadata != null && metadata.Count > 0)
        {
            var metaHtml = new StringBuilder();
            metaHtml.AppendLine("<div class='summary' style='margin-top: 20px;'>");
            metaHtml.AppendLine("<h3>Explanation Details</h3>");
            foreach (var kvp in metadata)
            {
                metaHtml.AppendLine($"<p><strong>{System.Net.WebUtility.HtmlEncode(kvp.Key)}:</strong> {System.Net.WebUtility.HtmlEncode(kvp.Value?.ToString() ?? "null")}</p>");
            }
            metaHtml.AppendLine("</div>");

            // Insert before closing container div
            html = html.Replace("</div>\n</body>", metaHtml.ToString() + "</div>\n</body>");
        }

        string filePath = GetSafeFilePath(filename, ".html");
        File.WriteAllText(filePath, html);

        return filePath;
    }

    /// <summary>
    /// Exports feature attributions to Markdown format.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="title">Document title.</param>
    /// <param name="metadata">Optional metadata to include.</param>
    /// <param name="filename">Output filename (without extension).</param>
    /// <returns>Path to the exported file.</returns>
    public string ExportToMarkdown(
        string[] featureNames,
        double[] attributions,
        string title = "Model Explanation",
        Dictionary<string, object>? metadata = null,
        string filename = "explanation")
    {
        var sb = new StringBuilder();

        sb.AppendLine($"# {title}");
        sb.AppendLine();
        sb.AppendLine($"*Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}*");
        sb.AppendLine();

        // Metadata
        if (metadata != null && metadata.Count > 0)
        {
            sb.AppendLine("## Details");
            sb.AppendLine();
            foreach (var kvp in metadata)
            {
                sb.AppendLine($"- **{kvp.Key}**: {kvp.Value}");
            }
            sb.AppendLine();
        }

        // Summary
        sb.AppendLine("## Summary Statistics");
        sb.AppendLine();
        sb.AppendLine($"- **Total Features**: {featureNames.Length}");
        sb.AppendLine($"- **Sum of Attributions**: {attributions.Sum():F4}");
        sb.AppendLine($"- **Mean Attribution**: {attributions.Average():F4}");
        sb.AppendLine($"- **Max Attribution**: {attributions.Max():F4}");
        sb.AppendLine($"- **Min Attribution**: {attributions.Min():F4}");
        sb.AppendLine();

        // Top positive contributors
        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => x.Attribution)
            .ToList();

        sb.AppendLine("## Top Positive Contributors");
        sb.AppendLine();
        sb.AppendLine("| Rank | Feature | Attribution |");
        sb.AppendLine("|------|---------|-------------|");
        int rank = 1;
        foreach (var (name, attr) in sorted.Where(x => x.Attribution > 0).Take(10))
        {
            sb.AppendLine($"| {rank++} | {name} | +{attr:F4} |");
        }
        sb.AppendLine();

        // Top negative contributors
        sb.AppendLine("## Top Negative Contributors");
        sb.AppendLine();
        sb.AppendLine("| Rank | Feature | Attribution |");
        sb.AppendLine("|------|---------|-------------|");
        rank = 1;
        foreach (var (name, attr) in sorted.Where(x => x.Attribution < 0).OrderBy(x => x.Attribution).Take(10))
        {
            sb.AppendLine($"| {rank++} | {name} | {attr:F4} |");
        }
        sb.AppendLine();

        // Full table
        sb.AppendLine("## All Features (by absolute attribution)");
        sb.AppendLine();
        sb.AppendLine("| Rank | Feature | Attribution | |Attribution| |");
        sb.AppendLine("|------|---------|-------------|-------------|");
        rank = 1;
        foreach (var (name, attr) in sorted.OrderByDescending(x => Math.Abs(x.Attribution)))
        {
            string sign = attr >= 0 ? "+" : "";
            sb.AppendLine($"| {rank++} | {name} | {sign}{attr:F4} | {Math.Abs(attr):F4} |");
        }

        string filePath = GetSafeFilePath(filename, ".md");
        File.WriteAllText(filePath, sb.ToString());

        return filePath;
    }

    /// <summary>
    /// Exports multiple explanations to a batch JSON file.
    /// </summary>
    /// <param name="explanations">List of explanations with instance IDs.</param>
    /// <param name="featureNames">Names of features (shared across all explanations).</param>
    /// <param name="metadata">Optional metadata.</param>
    /// <param name="filename">Output filename.</param>
    /// <returns>Path to the exported file.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you explain many instances at once (batch explanation),
    /// this method saves all of them in a single file for efficient storage and analysis.
    /// </para>
    /// </remarks>
    public string ExportBatchToJson(
        List<(string InstanceId, double[] Attributions)> explanations,
        string[] featureNames,
        Dictionary<string, object>? metadata = null,
        string filename = "batch_explanation")
    {
        var exportData = new JObject
        {
            ["timestamp"] = DateTime.UtcNow.ToString("o"),
            ["format_version"] = "1.0",
            ["batch_size"] = explanations.Count,
            ["feature_names"] = new JArray(featureNames)
        };

        if (metadata != null)
        {
            exportData["metadata"] = JObject.FromObject(metadata);
        }

        var instancesArray = new JArray();
        foreach (var (instanceId, attributions) in explanations)
        {
            var instanceObj = new JObject
            {
                ["instance_id"] = instanceId,
                ["attributions"] = new JArray(attributions)
            };
            instancesArray.Add(instanceObj);
        }
        exportData["instances"] = instancesArray;

        string filePath = GetSafeFilePath(filename, ".json");
        File.WriteAllText(filePath, exportData.ToString(Formatting.Indented));

        return filePath;
    }

    /// <summary>
    /// Exports a heatmap (2D attributions) to JSON format.
    /// </summary>
    /// <param name="attributions">2D attribution array.</param>
    /// <param name="metadata">Optional metadata.</param>
    /// <param name="filename">Output filename.</param>
    /// <returns>Path to the exported file.</returns>
    public string ExportHeatmapToJson(
        double[,] attributions,
        Dictionary<string, object>? metadata = null,
        string filename = "heatmap")
    {
        int height = attributions.GetLength(0);
        int width = attributions.GetLength(1);

        var exportData = new JObject
        {
            ["timestamp"] = DateTime.UtcNow.ToString("o"),
            ["format_version"] = "1.0",
            ["dimensions"] = new JObject
            {
                ["height"] = height,
                ["width"] = width
            }
        };

        if (metadata != null)
        {
            exportData["metadata"] = JObject.FromObject(metadata);
        }

        // Find statistics
        double min = double.MaxValue, max = double.MinValue, sum = 0;
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double val = attributions[h, w];
                min = Math.Min(min, val);
                max = Math.Max(max, val);
                sum += val;
            }
        }

        exportData["statistics"] = new JObject
        {
            ["min"] = min,
            ["max"] = max,
            ["mean"] = sum / (height * width),
            ["sum"] = sum
        };

        // Convert to 2D array
        var dataArray = new JArray();
        for (int h = 0; h < height; h++)
        {
            var rowArray = new JArray();
            for (int w = 0; w < width; w++)
            {
                rowArray.Add(attributions[h, w]);
            }
            dataArray.Add(rowArray);
        }
        exportData["data"] = dataArray;

        string filePath = GetSafeFilePath(filename, ".json");
        File.WriteAllText(filePath, exportData.ToString(Formatting.Indented));

        return filePath;
    }

    /// <summary>
    /// Exports a heatmap to an HTML page with visualization.
    /// </summary>
    public string ExportHeatmapToHtml(
        double[,] attributions,
        string title = "Attribution Heatmap",
        Dictionary<string, object>? metadata = null,
        string filename = "heatmap")
    {
        var sb = new StringBuilder();

        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang='en'>");
        sb.AppendLine("<head>");
        sb.AppendLine("    <meta charset='UTF-8'>");
        sb.AppendLine("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>");
        sb.AppendLine($"    <title>{System.Net.WebUtility.HtmlEncode(title)}</title>");
        sb.AppendLine("    <style>");
        sb.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }");
        sb.AppendLine("        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }");
        sb.AppendLine("        h1 { text-align: center; }");
        sb.AppendLine("        .summary { background: #e8f4f8; padding: 15px; border-radius: 4px; margin-bottom: 20px; }");
        sb.AppendLine("    </style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.AppendLine("    <div class='container'>");
        sb.AppendLine($"        <h1>{System.Net.WebUtility.HtmlEncode(title)}</h1>");

        // Metadata
        if (metadata != null && metadata.Count > 0)
        {
            sb.AppendLine("        <div class='summary'>");
            foreach (var kvp in metadata)
            {
                sb.AppendLine($"            <p><strong>{System.Net.WebUtility.HtmlEncode(kvp.Key)}:</strong> {System.Net.WebUtility.HtmlEncode(kvp.Value?.ToString() ?? "")}</p>");
            }
            sb.AppendLine("        </div>");
        }

        // Add heatmap visualization
        sb.AppendLine(_visualizer.GenerateHtmlHeatmap(attributions, "Spatial Attribution"));

        sb.AppendLine("    </div>");
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        string filePath = GetSafeFilePath(filename, ".html");
        File.WriteAllText(filePath, sb.ToString());

        return filePath;
    }

    /// <summary>
    /// Gets a safe file path, preventing path traversal attacks.
    /// </summary>
    private string GetSafeFilePath(string filename, string extension)
    {
        // Sanitize filename
        string sanitized = string.Join("_", filename.Split(Path.GetInvalidFileNameChars()));
        if (string.IsNullOrWhiteSpace(sanitized))
        {
            sanitized = "explanation";
        }

        // Ensure extension starts with dot
        if (!extension.StartsWith("."))
        {
            extension = "." + extension;
        }

        string fullPath = Path.Combine(_outputDirectory, sanitized + extension);

        // Verify path is within output directory
        string resolvedPath = Path.GetFullPath(fullPath);
        string resolvedDir = Path.GetFullPath(_outputDirectory);

        if (!resolvedPath.StartsWith(resolvedDir))
        {
            throw new InvalidOperationException("Invalid filename: path traversal attempt detected.");
        }

        return resolvedPath;
    }

    private static double CalculateStdDev(double[] values)
    {
        if (values.Length == 0) return 0;
        double mean = values.Average();
        double sumSquares = values.Sum(x => (x - mean) * (x - mean));
        return Math.Sqrt(sumSquares / values.Length);
    }
}
