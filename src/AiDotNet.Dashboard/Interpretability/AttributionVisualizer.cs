using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SysConsole = System.Console;

namespace AiDotNet.Dashboard.Interpretability;

/// <summary>
/// Visualizes feature attributions from interpretability explainers.
/// Supports console (ASCII) and HTML output formats.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> After computing feature attributions with SHAP, LIME, or other
/// explainers, this class helps you visualize the results. It can create:
/// - Bar charts showing which features are most important
/// - Heatmaps for spatial attributions (like GradCAM)
/// - Waterfall charts showing cumulative contributions
/// - Summary plots for global explanations
/// </para>
/// </remarks>
public class AttributionVisualizer
{
    private readonly int _width;
    private readonly int _height;
    private readonly bool _useColor;

    /// <summary>
    /// Default colors for positive and negative attributions.
    /// </summary>
    private static readonly System.ConsoleColor PositiveColor = System.ConsoleColor.Green;
    private static readonly System.ConsoleColor NegativeColor = System.ConsoleColor.Red;
    private static readonly System.ConsoleColor NeutralColor = System.ConsoleColor.Gray;

    /// <summary>
    /// Initializes a new attribution visualizer.
    /// </summary>
    /// <param name="width">Width of visualizations (default: 80 characters).</param>
    /// <param name="height">Height of visualizations (default: 20 lines).</param>
    /// <param name="useColor">Whether to use console colors (default: true).</param>
    public AttributionVisualizer(int width = 80, int height = 20, bool useColor = true)
    {
        _width = Math.Max(40, width);
        _height = Math.Max(10, height);
        _useColor = useColor;
    }

    /// <summary>
    /// Renders a horizontal bar chart of feature attributions to the console.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values for each feature.</param>
    /// <param name="title">Optional chart title.</param>
    /// <param name="topK">Show only top K features by absolute value (default: show all).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a bar chart where longer bars indicate
    /// more important features. Green bars push the prediction higher, red bars
    /// push it lower.
    /// </para>
    /// </remarks>
    public void RenderBarChart(
        string[] featureNames,
        double[] attributions,
        string? title = null,
        int? topK = null)
    {
        if (featureNames.Length != attributions.Length)
            throw new ArgumentException("Feature names and attributions must have the same length.");

        // Sort by absolute attribution and take top K
        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .Take(topK ?? featureNames.Length)
            .ToList();

        if (sorted.Count == 0)
        {
            SysConsole.WriteLine("No attributions to display.");
            return;
        }

        // Print title
        if (title is { Length: > 0 })
        {
            SysConsole.WriteLine();
            SysConsole.WriteLine(title);
            SysConsole.WriteLine(new string('=', title.Length));
        }

        // Calculate max values for scaling
        double maxAbs = sorted.Max(x => Math.Abs(x.Attribution));
        if (maxAbs < 1e-10) maxAbs = 1.0;

        int maxNameLength = sorted.Max(x => x.Name.Length);
        int barWidth = _width - maxNameLength - 15; // Leave space for name and value

        SysConsole.WriteLine();

        foreach (var (name, attribution) in sorted)
        {
            // Print feature name (right-aligned)
            SysConsole.Write(name.PadLeft(maxNameLength) + " ");

            // Calculate bar length
            int barLength = (int)(Math.Abs(attribution) / maxAbs * (barWidth / 2));
            int center = barWidth / 2;

            // Build bar string
            var bar = new char[barWidth];
            for (int i = 0; i < barWidth; i++) bar[i] = ' ';
            bar[center] = '|';

            if (attribution >= 0)
            {
                for (int i = center + 1; i <= center + barLength && i < barWidth; i++)
                    bar[i] = '#';
            }
            else
            {
                for (int i = center - 1; i >= center - barLength && i >= 0; i--)
                    bar[i] = '#';
            }

            // Print with color
            if (_useColor)
            {
                SysConsole.ForegroundColor = NeutralColor;
                for (int i = 0; i < center; i++)
                {
                    if (bar[i] == '#')
                    {
                        SysConsole.ForegroundColor = NegativeColor;
                        SysConsole.Write(bar[i]);
                        SysConsole.ForegroundColor = NeutralColor;
                    }
                    else
                    {
                        SysConsole.Write(bar[i]);
                    }
                }
                SysConsole.Write('|');
                for (int i = center + 1; i < barWidth; i++)
                {
                    if (bar[i] == '#')
                    {
                        SysConsole.ForegroundColor = PositiveColor;
                        SysConsole.Write(bar[i]);
                        SysConsole.ForegroundColor = NeutralColor;
                    }
                    else
                    {
                        SysConsole.Write(bar[i]);
                    }
                }
                SysConsole.ResetColor();
            }
            else
            {
                SysConsole.Write(new string(bar));
            }

            // Print value
            SysConsole.WriteLine($" {attribution,8:F4}");
        }

        SysConsole.WriteLine();
    }

    /// <summary>
    /// Renders a 2D heatmap of spatial attributions to the console.
    /// </summary>
    /// <param name="attributions">2D array of attributions [height, width].</param>
    /// <param name="title">Optional chart title.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is useful for visualizing GradCAM or other
    /// spatial attribution methods. Brighter areas indicate higher attribution.
    /// For images, this shows which regions the model is "looking at".
    /// </para>
    /// </remarks>
    public void RenderHeatmap(double[,] attributions, string? title = null)
    {
        int height = attributions.GetLength(0);
        int width = attributions.GetLength(1);

        // Print title
        if (title is { Length: > 0 })
        {
            SysConsole.WriteLine();
            SysConsole.WriteLine(title);
            SysConsole.WriteLine(new string('=', title.Length));
        }

        // Find min/max for normalization
        double min = double.MaxValue;
        double max = double.MinValue;
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                min = Math.Min(min, attributions[h, w]);
                max = Math.Max(max, attributions[h, w]);
            }
        }

        double range = max - min;
        if (range < 1e-10) range = 1.0;

        // Intensity characters from low to high
        char[] intensityChars = { ' ', '.', ':', '-', '=', '+', '*', '#', '@' };

        SysConsole.WriteLine();

        // Downsample if needed
        int stepH = Math.Max(1, height / _height);
        int stepW = Math.Max(1, width / (_width - 2));

        for (int h = 0; h < height; h += stepH)
        {
            SysConsole.Write("|");
            for (int w = 0; w < width; w += stepW)
            {
                // Average values in the cell
                double sum = 0;
                int count = 0;
                for (int dh = 0; dh < stepH && h + dh < height; dh++)
                {
                    for (int dw = 0; dw < stepW && w + dw < width; dw++)
                    {
                        sum += attributions[h + dh, w + dw];
                        count++;
                    }
                }
                double value = sum / count;

                // Normalize to 0-1
                double normalized = (value - min) / range;

                // Map to character
                int charIndex = Math.Min((int)(normalized * intensityChars.Length), intensityChars.Length - 1);

                if (_useColor)
                {
                    // Color based on intensity
                    if (normalized < 0.33)
                        SysConsole.ForegroundColor = System.ConsoleColor.Blue;
                    else if (normalized < 0.66)
                        SysConsole.ForegroundColor = System.ConsoleColor.Yellow;
                    else
                        SysConsole.ForegroundColor = System.ConsoleColor.Red;

                    SysConsole.Write(intensityChars[charIndex]);
                    SysConsole.ResetColor();
                }
                else
                {
                    SysConsole.Write(intensityChars[charIndex]);
                }
            }
            SysConsole.WriteLine("|");
        }

        // Print legend
        SysConsole.WriteLine();
        SysConsole.Write("Legend: ");
        for (int i = 0; i < intensityChars.Length; i++)
        {
            double value = min + (range * i / (intensityChars.Length - 1));
            SysConsole.Write($"'{intensityChars[i]}'={value:F2} ");
        }
        SysConsole.WriteLine();
    }

    /// <summary>
    /// Renders a waterfall chart showing cumulative attribution contributions.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values for each feature.</param>
    /// <param name="baseValue">Base value (expected value or baseline prediction).</param>
    /// <param name="title">Optional chart title.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A waterfall chart shows how each feature pushes the
    /// prediction up or down from a baseline. It's great for explaining individual
    /// predictions: "Starting from the average prediction, feature A added +0.3,
    /// feature B subtracted -0.1, etc."
    /// </para>
    /// </remarks>
    public void RenderWaterfallChart(
        string[] featureNames,
        double[] attributions,
        double baseValue,
        string? title = null)
    {
        if (featureNames.Length != attributions.Length)
            throw new ArgumentException("Feature names and attributions must have the same length.");

        // Print title
        if (title is { Length: > 0 })
        {
            SysConsole.WriteLine();
            SysConsole.WriteLine(title);
            SysConsole.WriteLine(new string('=', title.Length));
        }

        // Sort by absolute attribution (largest first)
        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .ToList();

        // Calculate cumulative values
        double cumulative = baseValue;
        double finalValue = baseValue + attributions.Sum();

        // Find range for scaling
        double minVal = Math.Min(baseValue, finalValue);
        double maxVal = Math.Max(baseValue, finalValue);
        foreach (var (_, attr) in sorted)
        {
            cumulative += attr;
            minVal = Math.Min(minVal, cumulative);
            maxVal = Math.Max(maxVal, cumulative);
        }

        double range = maxVal - minVal;
        if (range < 1e-10) range = 1.0;

        int maxNameLength = Math.Max(sorted.Max(x => x.Name.Length), 10);
        int chartWidth = _width - maxNameLength - 20;

        SysConsole.WriteLine();

        // Print base value
        int basePos = (int)((baseValue - minVal) / range * chartWidth);
        SysConsole.Write("Base Value".PadLeft(maxNameLength) + " ");
        SysConsole.Write(new string(' ', basePos));
        if (_useColor) SysConsole.ForegroundColor = System.ConsoleColor.Cyan;
        SysConsole.Write("|");
        SysConsole.ResetColor();
        SysConsole.WriteLine($" {baseValue,8:F4}");

        // Print each contribution
        cumulative = baseValue;
        foreach (var (name, attribution) in sorted)
        {
            double startVal = cumulative;
            cumulative += attribution;

            int startPos = (int)((startVal - minVal) / range * chartWidth);
            int endPos = (int)((cumulative - minVal) / range * chartWidth);

            SysConsole.Write(name.PadLeft(maxNameLength) + " ");

            // Draw the bar
            int left = Math.Min(startPos, endPos);
            int right = Math.Max(startPos, endPos);

            SysConsole.Write(new string(' ', left));

            if (_useColor)
                SysConsole.ForegroundColor = attribution >= 0 ? PositiveColor : NegativeColor;

            if (attribution >= 0)
            {
                SysConsole.Write("|");
                SysConsole.Write(new string('>', right - left - 1));
            }
            else
            {
                SysConsole.Write(new string('<', right - left - 1));
                SysConsole.Write("|");
            }

            SysConsole.ResetColor();
            SysConsole.Write(new string(' ', chartWidth - right));
            SysConsole.WriteLine($" {(attribution >= 0 ? "+" : "")}{attribution,7:F4} = {cumulative,8:F4}");
        }

        // Print final value
        int finalPos = (int)((finalValue - minVal) / range * chartWidth);
        SysConsole.Write("Prediction".PadLeft(maxNameLength) + " ");
        SysConsole.Write(new string(' ', finalPos));
        if (_useColor) SysConsole.ForegroundColor = System.ConsoleColor.Magenta;
        SysConsole.Write("|");
        SysConsole.ResetColor();
        SysConsole.WriteLine($" {finalValue,8:F4}");

        SysConsole.WriteLine();
    }

    /// <summary>
    /// Generates HTML for an interactive bar chart of attributions.
    /// Uses Chart.js for rendering.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values for each feature.</param>
    /// <param name="title">Chart title.</param>
    /// <param name="topK">Show only top K features.</param>
    /// <returns>HTML string containing the chart.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This generates HTML that can be embedded in a web page
    /// or saved to a file. The chart is interactive - you can hover over bars to
    /// see exact values.
    /// </para>
    /// </remarks>
    public string GenerateHtmlBarChart(
        string[] featureNames,
        double[] attributions,
        string title = "Feature Attributions",
        int? topK = null)
    {
        // Sort and take top K
        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .Take(topK ?? featureNames.Length)
            .Reverse() // Chart.js renders bottom-to-top for horizontal bars
            .ToList();

        var labels = string.Join(",", sorted.Select(x => $"'{EscapeJs(x.Name)}'"));
        var data = string.Join(",", sorted.Select(x => x.Attribution.ToString("F6")));
        var colors = string.Join(",", sorted.Select(x =>
            x.Attribution >= 0 ? "'rgba(75, 192, 92, 0.8)'" : "'rgba(255, 99, 132, 0.8)'"));

        string chartId = $"attr_chart_{Guid.NewGuid():N}";

        return $@"
<div style='width: 100%; max-width: 800px; margin: 20px auto;'>
    <canvas id='{chartId}'></canvas>
</div>
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
<script>
new Chart(document.getElementById('{chartId}'), {{
    type: 'bar',
    data: {{
        labels: [{labels}],
        datasets: [{{
            label: 'Attribution',
            data: [{data}],
            backgroundColor: [{colors}],
            borderColor: [{colors.Replace("0.8", "1")}],
            borderWidth: 1
        }}]
    }},
    options: {{
        indexAxis: 'y',
        responsive: true,
        plugins: {{
            title: {{
                display: true,
                text: '{EscapeJs(title)}'
            }},
            legend: {{
                display: false
            }}
        }},
        scales: {{
            x: {{
                beginAtZero: true,
                title: {{
                    display: true,
                    text: 'Attribution Value'
                }}
            }}
        }}
    }}
}});
</script>";
    }

    /// <summary>
    /// Generates HTML for a heatmap visualization.
    /// </summary>
    /// <param name="attributions">2D array of attributions.</param>
    /// <param name="title">Chart title.</param>
    /// <returns>HTML string containing the heatmap.</returns>
    public string GenerateHtmlHeatmap(double[,] attributions, string title = "Attribution Heatmap")
    {
        int height = attributions.GetLength(0);
        int width = attributions.GetLength(1);

        // Find min/max
        double min = double.MaxValue;
        double max = double.MinValue;
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                min = Math.Min(min, attributions[h, w]);
                max = Math.Max(max, attributions[h, w]);
            }
        }

        // Build data array
        var sb = new StringBuilder();
        sb.Append("[");
        for (int h = 0; h < height; h++)
        {
            if (h > 0) sb.Append(",");
            sb.Append("[");
            for (int w = 0; w < width; w++)
            {
                if (w > 0) sb.Append(",");
                sb.Append(attributions[h, w].ToString("F4"));
            }
            sb.Append("]");
        }
        sb.Append("]");

        string chartId = $"heatmap_{Guid.NewGuid():N}";

        return $@"
<div style='width: 100%; max-width: 800px; margin: 20px auto;'>
    <h3 style='text-align: center;'>{EscapeHtml(title)}</h3>
    <canvas id='{chartId}' width='{width * 10}' height='{height * 10}'></canvas>
    <div style='text-align: center; margin-top: 10px;'>
        <span style='background: linear-gradient(to right, blue, yellow, red); padding: 5px 50px;'></span>
        <br/>
        <span>{min:F2}</span> to <span>{max:F2}</span>
    </div>
</div>
<script>
(function() {{
    var canvas = document.getElementById('{chartId}');
    var ctx = canvas.getContext('2d');
    var data = {sb};
    var min = {min};
    var max = {max};
    var range = max - min || 1;
    var cellW = canvas.width / data[0].length;
    var cellH = canvas.height / data.length;

    for (var h = 0; h < data.length; h++) {{
        for (var w = 0; w < data[h].length; w++) {{
            var normalized = (data[h][w] - min) / range;
            var r = Math.floor(normalized * 255);
            var g = Math.floor((1 - Math.abs(normalized - 0.5) * 2) * 255);
            var b = Math.floor((1 - normalized) * 255);
            ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
            ctx.fillRect(w * cellW, h * cellH, cellW, cellH);
        }}
    }}
}})();
</script>";
    }

    /// <summary>
    /// Generates a complete HTML page with attribution visualizations.
    /// </summary>
    /// <param name="featureNames">Names of features.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="title">Page title.</param>
    /// <param name="baseValue">Optional base value for waterfall chart.</param>
    /// <returns>Complete HTML document.</returns>
    public string GenerateHtmlPage(
        string[] featureNames,
        double[] attributions,
        string title = "Attribution Explanation",
        double? baseValue = null)
    {
        var barChart = GenerateHtmlBarChart(featureNames, attributions, "Feature Attributions", 15);

        var sb = new StringBuilder();
        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang='en'>");
        sb.AppendLine("<head>");
        sb.AppendLine("    <meta charset='UTF-8'>");
        sb.AppendLine("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>");
        sb.AppendLine($"    <title>{EscapeHtml(title)}</title>");
        sb.AppendLine("    <style>");
        sb.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }");
        sb.AppendLine("        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }");
        sb.AppendLine("        h1 { color: #333; text-align: center; }");
        sb.AppendLine("        .summary { background: #e8f4f8; padding: 15px; border-radius: 4px; margin-bottom: 20px; }");
        sb.AppendLine("        .positive { color: #2e7d32; }");
        sb.AppendLine("        .negative { color: #c62828; }");
        sb.AppendLine("        table { width: 100%; border-collapse: collapse; margin-top: 20px; }");
        sb.AppendLine("        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }");
        sb.AppendLine("        th { background: #f0f0f0; }");
        sb.AppendLine("    </style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.AppendLine("    <div class='container'>");
        sb.AppendLine($"        <h1>{EscapeHtml(title)}</h1>");

        // Summary section
        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .ToList();

        double totalPositive = sorted.Where(x => x.Attribution > 0).Sum(x => x.Attribution);
        double totalNegative = sorted.Where(x => x.Attribution < 0).Sum(x => x.Attribution);

        sb.AppendLine("        <div class='summary'>");
        sb.AppendLine($"            <p><strong>Total Features:</strong> {featureNames.Length}</p>");
        sb.AppendLine($"            <p><strong>Total Positive Contribution:</strong> <span class='positive'>+{totalPositive:F4}</span></p>");
        sb.AppendLine($"            <p><strong>Total Negative Contribution:</strong> <span class='negative'>{totalNegative:F4}</span></p>");
        if (baseValue.HasValue)
        {
            sb.AppendLine($"            <p><strong>Base Value:</strong> {baseValue.Value:F4}</p>");
            sb.AppendLine($"            <p><strong>Prediction:</strong> {baseValue.Value + attributions.Sum():F4}</p>");
        }
        sb.AppendLine("        </div>");

        // Bar chart
        sb.AppendLine(barChart);

        // Table of all attributions
        sb.AppendLine("        <h2>All Feature Attributions</h2>");
        sb.AppendLine("        <table>");
        sb.AppendLine("            <tr><th>Rank</th><th>Feature</th><th>Attribution</th><th>|Attribution|</th></tr>");

        int rank = 1;
        foreach (var (name, attr) in sorted)
        {
            string cssClass = attr >= 0 ? "positive" : "negative";
            sb.AppendLine($"            <tr><td>{rank++}</td><td>{EscapeHtml(name)}</td><td class='{cssClass}'>{attr:F6}</td><td>{Math.Abs(attr):F6}</td></tr>");
        }

        sb.AppendLine("        </table>");
        sb.AppendLine("    </div>");
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        return sb.ToString();
    }

    private static string EscapeJs(string s) => s.Replace("\\", "\\\\").Replace("'", "\\'").Replace("\n", "\\n");
    private static string EscapeHtml(string s) => System.Net.WebUtility.HtmlEncode(s);
}
