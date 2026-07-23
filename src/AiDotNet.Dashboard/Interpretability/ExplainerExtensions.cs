using System;
using System.Collections.Generic;
using System.Linq;
using SysConsole = System.Console;

namespace AiDotNet.Dashboard.Interpretability;

/// <summary>
/// Extension methods for easily visualizing interpretability results.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These extension methods let you quickly visualize explanation
/// results without manually extracting data. Just call .Visualize() on your explanation
/// object.
/// </para>
/// </remarks>
public static class ExplainerExtensions
{
    /// <summary>
    /// Visualizes feature attributions to the console.
    /// </summary>
    /// <param name="featureNames">Feature names.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="title">Optional title.</param>
    /// <param name="topK">Number of top features to show.</param>
    public static void VisualizeAttributions(
        string[] featureNames,
        double[] attributions,
        string? title = null,
        int topK = 15)
    {
        var visualizer = new AttributionVisualizer();
        visualizer.RenderBarChart(featureNames, attributions, title ?? "Feature Attributions", topK);
    }

    /// <summary>
    /// Visualizes a 2D heatmap to the console.
    /// </summary>
    /// <param name="heatmap">2D attribution values.</param>
    /// <param name="title">Optional title.</param>
    public static void VisualizeHeatmap(double[,] heatmap, string? title = null)
    {
        var visualizer = new AttributionVisualizer();
        visualizer.RenderHeatmap(heatmap, title ?? "Attribution Heatmap");
    }

    /// <summary>
    /// Visualizes attributions as a waterfall chart.
    /// </summary>
    /// <param name="featureNames">Feature names.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="baseValue">Base/expected value.</param>
    /// <param name="title">Optional title.</param>
    public static void VisualizeWaterfall(
        string[] featureNames,
        double[] attributions,
        double baseValue,
        string? title = null)
    {
        var visualizer = new AttributionVisualizer();
        visualizer.RenderWaterfallChart(featureNames, attributions, baseValue, title ?? "Attribution Waterfall");
    }

    /// <summary>
    /// Generates an HTML visualization of attributions.
    /// </summary>
    /// <param name="featureNames">Feature names.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="title">Chart title.</param>
    /// <param name="topK">Number of top features.</param>
    /// <returns>HTML string.</returns>
    public static string ToHtml(
        string[] featureNames,
        double[] attributions,
        string title = "Feature Attributions",
        int topK = 15)
    {
        var visualizer = new AttributionVisualizer();
        return visualizer.GenerateHtmlPage(featureNames, attributions, title);
    }

    /// <summary>
    /// Saves attributions to an HTML file.
    /// </summary>
    /// <param name="featureNames">Feature names.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="filePath">Output file path.</param>
    /// <param name="title">Chart title.</param>
    public static void SaveToHtml(
        string[] featureNames,
        double[] attributions,
        string filePath,
        string title = "Feature Attributions")
    {
        string html = ToHtml(featureNames, attributions, title);
        System.IO.File.WriteAllText(filePath, html);
    }

    /// <summary>
    /// Creates a quick summary string of attributions.
    /// </summary>
    /// <param name="featureNames">Feature names.</param>
    /// <param name="attributions">Attribution values.</param>
    /// <param name="topK">Number of top features to include.</param>
    /// <returns>Summary string.</returns>
    public static string Summarize(
        string[] featureNames,
        double[] attributions,
        int topK = 5)
    {
        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .Take(topK)
            .ToList();

        var lines = new List<string>
        {
            $"Top {topK} Features by Attribution:"
        };

        foreach (var (name, attr) in sorted)
        {
            string sign = attr >= 0 ? "+" : "";
            lines.Add($"  {name}: {sign}{attr:F4}");
        }

        double total = attributions.Sum();
        lines.Add($"Total Attribution: {total:F4}");

        return string.Join(Environment.NewLine, lines);
    }

    /// <summary>
    /// Compares two sets of attributions and shows differences.
    /// </summary>
    /// <param name="featureNames">Feature names.</param>
    /// <param name="attributions1">First set of attributions.</param>
    /// <param name="attributions2">Second set of attributions.</param>
    /// <param name="label1">Label for first set.</param>
    /// <param name="label2">Label for second set.</param>
    public static void CompareAttributions(
        string[] featureNames,
        double[] attributions1,
        double[] attributions2,
        string label1 = "Explanation 1",
        string label2 = "Explanation 2")
    {
        SysConsole.WriteLine();
        SysConsole.WriteLine("Attribution Comparison");
        SysConsole.WriteLine(new string('=', 70));
        SysConsole.WriteLine();
        SysConsole.WriteLine($"{"Feature",-25} {label1,15} {label2,15} {"Difference",12}");
        SysConsole.WriteLine(new string('-', 70));

        var differences = featureNames
            .Select((name, i) => (
                Name: name,
                Attr1: i < attributions1.Length ? attributions1[i] : 0,
                Attr2: i < attributions2.Length ? attributions2[i] : 0))
            .Select(x => (x.Name, x.Attr1, x.Attr2, Diff: x.Attr2 - x.Attr1))
            .OrderByDescending(x => Math.Abs(x.Diff))
            .ToList();

        foreach (var (name, attr1, attr2, diff) in differences.Take(15))
        {
            SysConsole.ForegroundColor = diff > 0 ? System.ConsoleColor.Green : (diff < 0 ? System.ConsoleColor.Red : System.ConsoleColor.Gray);
            string diffStr = diff >= 0 ? $"+{diff:F4}" : $"{diff:F4}";
            SysConsole.WriteLine($"{name,-25} {attr1,15:F4} {attr2,15:F4} {diffStr,12}");
            SysConsole.ResetColor();
        }

        SysConsole.WriteLine();
        SysConsole.WriteLine($"Correlation: {CalculateCorrelation(attributions1, attributions2):F4}");
        SysConsole.WriteLine();
    }

    /// <summary>
    /// Prints a formatted table of attributions.
    /// </summary>
    public static void PrintAttributionTable(
        string[] featureNames,
        double[] attributions,
        string title = "Feature Attributions")
    {
        SysConsole.WriteLine();
        SysConsole.WriteLine(title);
        SysConsole.WriteLine(new string('=', 60));
        SysConsole.WriteLine();
        SysConsole.WriteLine($"{"Rank",5} {"Feature",-30} {"Attribution",15} {"Direction",10}");
        SysConsole.WriteLine(new string('-', 60));

        var sorted = featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .Select((x, i) => (Rank: i + 1, x.Name, x.Attribution))
            .ToList();

        foreach (var (rank, name, attr) in sorted)
        {
            string direction = attr >= 0 ? "+" : "-";
            SysConsole.ForegroundColor = attr >= 0 ? System.ConsoleColor.Green : System.ConsoleColor.Red;
            SysConsole.WriteLine($"{rank,5} {name,-30} {attr,15:F6} {direction,10}");
            SysConsole.ResetColor();
        }

        SysConsole.WriteLine();
        SysConsole.WriteLine($"Sum of Attributions: {attributions.Sum():F6}");
        SysConsole.WriteLine($"Mean Attribution: {attributions.Average():F6}");
        SysConsole.WriteLine($"Std Dev: {CalculateStdDev(attributions):F6}");
        SysConsole.WriteLine();
    }

    /// <summary>
    /// Gets the top K features by attribution magnitude.
    /// </summary>
    public static (string Name, double Attribution)[] GetTopFeatures(
        string[] featureNames,
        double[] attributions,
        int k = 10)
    {
        return featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .Take(k)
            .ToArray();
    }

    /// <summary>
    /// Gets features that exceed a threshold.
    /// </summary>
    public static (string Name, double Attribution)[] GetSignificantFeatures(
        string[] featureNames,
        double[] attributions,
        double threshold = 0.01)
    {
        return featureNames
            .Zip(attributions, (name, attr) => (Name: name, Attribution: attr))
            .Where(x => Math.Abs(x.Attribution) >= threshold)
            .OrderByDescending(x => Math.Abs(x.Attribution))
            .ToArray();
    }

    private static double CalculateCorrelation(double[] x, double[] y)
    {
        if (x.Length != y.Length || x.Length == 0) return 0;

        double meanX = x.Average();
        double meanY = y.Average();

        double numerator = 0;
        double sumSqX = 0;
        double sumSqY = 0;

        for (int i = 0; i < x.Length; i++)
        {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            numerator += dx * dy;
            sumSqX += dx * dx;
            sumSqY += dy * dy;
        }

        double denominator = Math.Sqrt(sumSqX * sumSqY);
        return denominator < 1e-10 ? 0 : numerator / denominator;
    }

    private static double CalculateStdDev(double[] values)
    {
        if (values.Length == 0) return 0;
        double mean = values.Average();
        double sumSquares = values.Sum(x => (x - mean) * (x - mean));
        return Math.Sqrt(sumSquares / values.Length);
    }
}
