// Copyright (c) 2024 AiDotNet. All rights reserved.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.Tensors.Engines.DirectGpu.Profiling;

/// <summary>
/// Exports profiling results to various formats (JSON, CSV, Markdown).
/// </summary>
public static class ProfileExporter
{
    /// <summary>
    /// Exports a ProfileResult to JSON format.
    /// </summary>
    /// <param name="result">The profile result to export.</param>
    /// <param name="indented">Whether to format with indentation.</param>
    /// <returns>JSON string representation.</returns>
    public static string ToJson(ProfileResult result, bool indented = true)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = indented,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter() }
        };

        return JsonSerializer.Serialize(result, options);
    }

    /// <summary>
    /// Exports a ProfileResult to JSON and writes to file.
    /// </summary>
    /// <param name="result">The profile result to export.</param>
    /// <param name="filePath">Output file path.</param>
    /// <param name="indented">Whether to format with indentation.</param>
    public static void ToJsonFile(ProfileResult result, string filePath, bool indented = true)
    {
        var json = ToJson(result, indented);
        File.WriteAllText(filePath, json, Encoding.UTF8);
    }

    /// <summary>
    /// Exports a ProfileResult to CSV format.
    /// </summary>
    /// <param name="result">The profile result to export.</param>
    /// <param name="includeHeader">Whether to include column headers.</param>
    /// <returns>CSV string representation.</returns>
    public static string ToCsv(ProfileResult result, bool includeHeader = true)
    {
        var sb = new StringBuilder();

        if (includeHeader)
        {
            sb.AppendLine("M,N,K,GFLOPS,EfficiencyPct,MemoryBWGBs,ArithmeticIntensity," +
                          "RooflineLimitGFLOPS,QueueTimeUs,SubmitToStartUs,ExecutionTimeUs," +
                          "TotalTimeUs,LaunchOverheadPct,Bottleneck,RecommendedAction," +
                          "OccupancyPct,LimitingFactor");
        }

        foreach (var entry in result.Entries)
        {
            var fields = new string[]
            {
                entry.M.ToString(CultureInfo.InvariantCulture),
                entry.N.ToString(CultureInfo.InvariantCulture),
                entry.K.ToString(CultureInfo.InvariantCulture),
                entry.Gflops.ToString("F2", CultureInfo.InvariantCulture),
                entry.EfficiencyPercent.ToString("F2", CultureInfo.InvariantCulture),
                entry.MemoryBandwidthGBs.ToString("F2", CultureInfo.InvariantCulture),
                entry.ArithmeticIntensity.ToString("F4", CultureInfo.InvariantCulture),
                entry.RooflineLimitGflops.ToString("F2", CultureInfo.InvariantCulture),
                entry.QueueTimeUs.ToString("F2", CultureInfo.InvariantCulture),
                entry.SubmitToStartUs.ToString("F2", CultureInfo.InvariantCulture),
                entry.ExecutionTimeUs.ToString("F2", CultureInfo.InvariantCulture),
                entry.TotalTimeUs.ToString("F2", CultureInfo.InvariantCulture),
                entry.LaunchOverheadPercent.ToString("F2", CultureInfo.InvariantCulture),
                entry.Bottleneck.ToString(),
                entry.RecommendedAction.ToString(),
                entry.Occupancy?.TheoreticalOccupancy.ToString("F1", CultureInfo.InvariantCulture) ?? "",
                entry.Occupancy?.LimitingFactor.ToString() ?? ""
            };

            sb.AppendLine(string.Join(",", fields));
        }

        return sb.ToString();
    }

    /// <summary>
    /// Exports a ProfileResult to CSV and writes to file.
    /// </summary>
    /// <param name="result">The profile result to export.</param>
    /// <param name="filePath">Output file path.</param>
    /// <param name="includeHeader">Whether to include column headers.</param>
    public static void ToCsvFile(ProfileResult result, string filePath, bool includeHeader = true)
    {
        var csv = ToCsv(result, includeHeader);
        File.WriteAllText(filePath, csv, Encoding.UTF8);
    }

    /// <summary>
    /// Exports a ProfileResult to Markdown table format.
    /// </summary>
    /// <param name="result">The profile result to export.</param>
    /// <returns>Markdown string representation.</returns>
    public static string ToMarkdown(ProfileResult result)
    {
        var sb = new StringBuilder();

        sb.AppendLine("# GPU GEMM Profiling Report");
        sb.AppendLine();
        sb.AppendLine("## Device Information");
        sb.AppendLine();
        sb.AppendLine($"- **Device**: {result.DeviceName}");
        if (result.Architecture != null)
        {
            sb.AppendLine($"- **Architecture**: {result.Architecture.Name}");
            sb.AppendLine($"- **Wavefront Size**: {result.Architecture.WavefrontSize}");
            sb.AppendLine($"- **VGPRs/SIMD**: {result.Architecture.VgprsPerSimd}");
            sb.AppendLine($"- **LDS/CU**: {result.Architecture.LdsPerCuBytes / 1024} KB");
        }
        sb.AppendLine($"- **Peak GFLOPS**: {result.PeakGflops:F0}");
        sb.AppendLine($"- **Peak Bandwidth**: {result.PeakBandwidthGBs:F0} GB/s");
        sb.AppendLine($"- **Ridge Point**: {result.RidgePoint:F1} FLOPS/byte");
        sb.AppendLine();

        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine($"- **Best GFLOPS**: {result.BestGflops:F0}");
        sb.AppendLine($"- **Best Efficiency**: {result.BestEfficiencyPercent:F1}%");
        var (bm, bn, bk) = result.BestSize;
        sb.AppendLine($"- **Best Size**: {bm}x{bn}x{bk}");
        sb.AppendLine($"- **Profile Duration**: {result.ProfileDurationSeconds:F1}s");
        sb.AppendLine();

        sb.AppendLine("## Performance Results");
        sb.AppendLine();
        sb.AppendLine("| Size | GFLOPS | Efficiency | BW (GB/s) | AI (F/B) | Bottleneck | Action |");
        sb.AppendLine("|------|--------|------------|-----------|----------|------------|--------|");

        foreach (var entry in result.Entries.OrderBy(e => e.M * e.N * e.K))
        {
            string sizeStr = entry.M == entry.N && entry.N == entry.K
                ? entry.M.ToString()
                : $"{entry.M}x{entry.N}x{entry.K}";

            sb.AppendLine($"| {sizeStr} | {entry.Gflops:F0} | {entry.EfficiencyPercent:F1}% | " +
                          $"{entry.MemoryBandwidthGBs:F1} | {entry.ArithmeticIntensity:F2} | " +
                          $"{entry.Bottleneck} | {entry.RecommendedAction} |");
        }

        sb.AppendLine();
        sb.AppendLine("## Bottleneck Analysis");
        sb.AppendLine();

        var groups = result.GetBottleneckGroups();
        foreach (var group in groups.OrderByDescending(g => g.Value.Count))
        {
            sb.AppendLine($"### {group.Key}");
            sb.AppendLine();
            sb.AppendLine($"- **Count**: {group.Value.Count} sizes");
            sb.AppendLine($"- **Sizes**: {string.Join(", ", group.Value.Select(e => e.M.ToString()))}");
            sb.AppendLine();
        }

        sb.AppendLine("---");
        sb.AppendLine($"*Generated at {result.ProfileStartTime:yyyy-MM-dd HH:mm:ss}*");

        return sb.ToString();
    }

    /// <summary>
    /// Exports a ProfileResult to Markdown and writes to file.
    /// </summary>
    /// <param name="result">The profile result to export.</param>
    /// <param name="filePath">Output file path.</param>
    public static void ToMarkdownFile(ProfileResult result, string filePath)
    {
        var markdown = ToMarkdown(result);
        File.WriteAllText(filePath, markdown, Encoding.UTF8);
    }

    /// <summary>
    /// Exports only the roofline curve data to CSV for plotting.
    /// </summary>
    /// <param name="analyzer">The roofline analyzer.</param>
    /// <param name="entries">Optional profile entries to include as data points.</param>
    /// <returns>CSV string with roofline curve and data points.</returns>
    public static string RooflineToCsv(RooflineAnalyzer analyzer, IEnumerable<GemmProfileEntry>? entries = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine("Type,ArithmeticIntensity,GFLOPS,Label");

        // Generate roofline curve points
        var curve = analyzer.GenerateRooflineCurve(0.1, 100, 100);
        foreach (var (ai, gflops) in curve)
        {
            sb.AppendLine($"Roofline,{ai.ToString("F4", CultureInfo.InvariantCulture)}," +
                          $"{gflops.ToString("F2", CultureInfo.InvariantCulture)},");
        }

        // Add ridge point
        sb.AppendLine($"RidgePoint,{analyzer.RidgePoint.ToString("F4", CultureInfo.InvariantCulture)}," +
                      $"{analyzer.PeakGflops.ToString("F2", CultureInfo.InvariantCulture)},Ridge");

        // Add data points from profiling
        if (entries != null)
        {
            foreach (var entry in entries)
            {
                sb.AppendLine($"DataPoint,{entry.ArithmeticIntensity.ToString("F4", CultureInfo.InvariantCulture)}," +
                              $"{entry.Gflops.ToString("F2", CultureInfo.InvariantCulture)}," +
                              $"{entry.M}x{entry.N}x{entry.K}");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Exports compact summary for quick reference.
    /// </summary>
    /// <param name="result">The profile result to export.</param>
    /// <returns>Compact summary string.</returns>
    public static string ToCompactSummary(ProfileResult result)
    {
        var sb = new StringBuilder();

        sb.AppendLine($"GPU: {result.DeviceName}");
        sb.AppendLine($"Peak: {result.PeakGflops:F0} GFLOPS | {result.PeakBandwidthGBs:F0} GB/s");
        sb.AppendLine($"Best: {result.BestGflops:F0} GFLOPS ({result.BestEfficiencyPercent:F1}%)");
        sb.AppendLine();

        // Show size categories
        var smallSizes = result.Entries.Where(e => e.M <= 512).ToList();
        var mediumSizes = result.Entries.Where(e => e.M > 512 && e.M <= 2048).ToList();
        var largeSizes = result.Entries.Where(e => e.M > 2048).ToList();

        if (smallSizes.Count > 0)
        {
            sb.AppendLine($"Small (<=512): {smallSizes.Average(e => e.Gflops):F0} avg GFLOPS, " +
                          $"typical: {smallSizes.GroupBy(e => e.Bottleneck).OrderByDescending(g => g.Count()).First().Key}");
        }

        if (mediumSizes.Count > 0)
        {
            sb.AppendLine($"Medium (513-2048): {mediumSizes.Average(e => e.Gflops):F0} avg GFLOPS, " +
                          $"typical: {mediumSizes.GroupBy(e => e.Bottleneck).OrderByDescending(g => g.Count()).First().Key}");
        }

        if (largeSizes.Count > 0)
        {
            sb.AppendLine($"Large (>2048): {largeSizes.Average(e => e.Gflops):F0} avg GFLOPS, " +
                          $"typical: {largeSizes.GroupBy(e => e.Bottleneck).OrderByDescending(g => g.Count()).First().Key}");
        }

        return sb.ToString();
    }
}
