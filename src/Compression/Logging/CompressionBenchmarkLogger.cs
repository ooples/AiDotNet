using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Factories;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using System.Linq;
using System;

namespace AiDotNet.Compression.Logging;

/// <summary>
/// Logger for recording and analyzing model compression benchmarks.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This specialized logger tracks detailed metrics about model 
/// compression performance, allowing you to compare different compression techniques
/// and configurations. It can generate reports to help you choose the best approach
/// for your specific models and requirements.
/// </para>
/// </remarks>
public class CompressionBenchmarkLogger
{
    private readonly ILogging _logger = default!;
    private readonly string _benchmarkDirectory;
    private readonly List<CompressionBenchmarkResult> _benchmarkResults = new();
    
    /// <summary>
    /// Initializes a new instance of the CompressionBenchmarkLogger class.
    /// </summary>
    /// <param name="options">Logging configuration options.</param>
    /// <param name="benchmarkDirectory">Directory where benchmark results will be stored.</param>
    public CompressionBenchmarkLogger(LoggingOptions options, string benchmarkDirectory = "Benchmarks")
    {
        LoggingFactory.Configure(options);
        _logger = LoggingFactory.GetContextualLogger("Component", "CompressionBenchmark");
        
        _benchmarkDirectory = benchmarkDirectory;
        Directory.CreateDirectory(_benchmarkDirectory);
        
        _logger.Information("Compression benchmark logger initialized. Results will be stored in {Directory}", benchmarkDirectory);
    }
    
    /// <summary>
    /// Records the result of a compression benchmark.
    /// </summary>
    /// <param name="result">The benchmark result to record.</param>
    public void RecordBenchmarkResult(CompressionBenchmarkResult result)
    {
        _benchmarkResults.Add(result);
        
        _logger.Information(
            "Recorded benchmark result for {Technique} on {Model}. " +
            "Compression ratio: {Ratio:P2}, Accuracy impact: {Accuracy:P2}, " +
            "Inference speedup: {Speedup:F2}x, Memory reduction: {MemoryReduction:P2}",
            result.CompressionTechnique,
            result.ModelName,
            result.CompressionRatio,
            result.AccuracyImpact,
            result.InferenceSpeedup,
            result.MemoryReduction);
            
        // Save the individual result to a file
        var resultFileName = $"{result.ModelName}_{result.CompressionTechnique}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        var resultFilePath = Path.Combine(_benchmarkDirectory, resultFileName);
        
        File.WriteAllText(resultFilePath, JsonConvert.SerializeObject(result, Formatting.Indented));
    }
    
    /// <summary>
    /// Generates a summary report of all benchmark results.
    /// </summary>
    /// <param name="outputPath">Path where the report should be saved.</param>
    /// <returns>Path to the generated report file.</returns>
    public string GenerateBenchmarkReport(string? outputPath = null)
    {
        if (_benchmarkResults.Count == 0)
        {
            _logger.Warning("No benchmark results available to generate report");
            return string.Empty;
        }
        
        // Default output path if none provided
        outputPath ??= Path.Combine(_benchmarkDirectory, $"compression_benchmark_report_{DateTime.Now:yyyyMMdd_HHmmss}.md");
        
        var reportBuilder = new StringBuilder();
        reportBuilder.AppendLine("# Model Compression Benchmark Report");
        reportBuilder.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        reportBuilder.AppendLine();
        
        // Group results by model name for comparison
        var resultsByModel = _benchmarkResults.GroupBy(r => r.ModelName).ToList();
        
        foreach (var modelGroup in resultsByModel)
        {
            reportBuilder.AppendLine($"## Model: {modelGroup.Key}");
            reportBuilder.AppendLine();
            
            // Create comparison table
            reportBuilder.AppendLine("| Compression Technique | Ratio | Size Reduction | Accuracy Impact | Inference Speedup | Memory Reduction |");
            reportBuilder.AppendLine("|---------------------|-------|---------------|----------------|-----------------|-----------------|");
            
            foreach (var result in modelGroup.OrderByDescending(r => r.InferenceSpeedup))
            {
                reportBuilder.AppendLine(
                    $"| {result.CompressionTechnique} | " +
                    $"{result.CompressionRatio:P2} | " +
                    $"{(1 - 1/result.CompressionRatio):P2} | " +
                    $"{result.AccuracyImpact:P2} | " +
                    $"{result.InferenceSpeedup:F2}x | " +
                    $"{result.MemoryReduction:P2} |");
            }
            
            reportBuilder.AppendLine();
            
            // Add recommendations section
            var bestOverall = modelGroup.OrderByDescending(r => 
                r.InferenceSpeedup * (1 - Math.Abs(r.AccuracyImpact)) * r.CompressionRatio
            ).First();
            
            var bestSize = modelGroup.OrderByDescending(r => r.CompressionRatio).First();
            var bestSpeed = modelGroup.OrderByDescending(r => r.InferenceSpeedup).First();
            var bestAccuracy = modelGroup.OrderBy(r => Math.Abs(r.AccuracyImpact)).First();
            
            reportBuilder.AppendLine("### Recommendations");
            reportBuilder.AppendLine();
            reportBuilder.AppendLine($"- **Best Overall Balance**: {bestOverall.CompressionTechnique}");
            reportBuilder.AppendLine($"- **Best Size Reduction**: {bestSize.CompressionTechnique} ({bestSize.CompressionRatio:P2})");
            reportBuilder.AppendLine($"- **Best Inference Speed**: {bestSpeed.CompressionTechnique} ({bestSpeed.InferenceSpeedup:F2}x)");
            reportBuilder.AppendLine($"- **Best Accuracy Preservation**: {bestAccuracy.CompressionTechnique} ({bestAccuracy.AccuracyImpact:P2})");
            reportBuilder.AppendLine();
            
            // Add detailed notes if available
            if (modelGroup.Any(r => !string.IsNullOrEmpty(r.Notes)))
            {
                reportBuilder.AppendLine("### Detailed Notes");
                reportBuilder.AppendLine();
                
                foreach (var result in modelGroup)
                {
                    if (!string.IsNullOrEmpty(result.Notes))
                    {
                        reportBuilder.AppendLine($"#### {result.CompressionTechnique}");
                        reportBuilder.AppendLine(result.Notes);
                        reportBuilder.AppendLine();
                    }
                }
            }
        }
        
        // Add cross-model analysis
        if (resultsByModel.Count > 1)
        {
            reportBuilder.AppendLine("## Cross-Model Analysis");
            reportBuilder.AppendLine();
            
            // Find which techniques work best across models
            var techniquePerformance = _benchmarkResults
                .GroupBy(r => r.CompressionTechnique)
                .Select(g => new
                {
                    Technique = g.Key,
                    AverageRatio = g.Average(r => r.CompressionRatio),
                    AverageAccuracyImpact = g.Average(r => Math.Abs(r.AccuracyImpact)),
                    AverageSpeedup = g.Average(r => r.InferenceSpeedup)
                })
                .OrderByDescending(t => t.AverageSpeedup * (1 - t.AverageAccuracyImpact) * t.AverageRatio)
                .ToList();
                
            reportBuilder.AppendLine("| Technique | Avg. Ratio | Avg. Accuracy Impact | Avg. Speedup |");
            reportBuilder.AppendLine("|-----------|------------|----------------------|-------------|");
            
            foreach (var technique in techniquePerformance)
            {
                reportBuilder.AppendLine(
                    $"| {technique.Technique} | " +
                    $"{technique.AverageRatio:P2} | " +
                    $"{technique.AverageAccuracyImpact:P2} | " +
                    $"{technique.AverageSpeedup:F2}x |");
            }
            
            reportBuilder.AppendLine();
            reportBuilder.AppendLine("### General Recommendations");
            reportBuilder.AppendLine();
            reportBuilder.AppendLine($"- **Best Overall Technique**: {techniquePerformance.First().Technique}");
            reportBuilder.AppendLine($"- **Most Consistent**: {_benchmarkResults.GroupBy(r => r.CompressionTechnique).OrderBy(g => g.Select(r => r.AccuracyImpact).StdDev()).First().Key}");
        }
        
        // Write the report to the file
        File.WriteAllText(outputPath, reportBuilder.ToString());
        
        _logger.Information("Generated compression benchmark report at {ReportPath}", outputPath);
        
        return outputPath;
    }
    
    /// <summary>
    /// Saves all benchmark results to a JSON file.
    /// </summary>
    /// <param name="outputPath">Path where the JSON file should be saved.</param>
    /// <returns>Path to the generated JSON file.</returns>
    public string SaveBenchmarkResults(string? outputPath = null)
    {
        outputPath ??= Path.Combine(_benchmarkDirectory, $"compression_benchmark_results_{DateTime.Now:yyyyMMdd_HHmmss}.json");
        
        File.WriteAllText(outputPath, JsonConvert.SerializeObject(_benchmarkResults, Formatting.Indented));
        
        _logger.Information("Saved {Count} benchmark results to {FilePath}", _benchmarkResults.Count, outputPath);
        
        return outputPath;
    }
    
    /// <summary>
    /// Loads benchmark results from a JSON file.
    /// </summary>
    /// <param name="filePath">Path to the JSON file containing benchmark results.</param>
    /// <returns>Number of benchmark results loaded.</returns>
    public int LoadBenchmarkResults(string filePath)
    {
        if (!File.Exists(filePath))
        {
            _logger.Error("Benchmark results file not found: {FilePath}", filePath);
            return 0;
        }
        
        try
        {
            var json = File.ReadAllText(filePath);
            var results = JsonConvert.DeserializeObject<List<CompressionBenchmarkResult>>(json);
            
            if (results != null)
            {
                _benchmarkResults.AddRange(results);
                _logger.Information("Loaded {Count} benchmark results from {FilePath}", results.Count, filePath);
            }
            else
            {
                _logger.Warning("No benchmark results found in {FilePath}", filePath);
                return 0;
            }
            
            return results?.Count ?? 0;
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Failed to load benchmark results from {FilePath}", filePath);
            return 0;
        }
    }
    
    /// <summary>
    /// Clears all recorded benchmark results.
    /// </summary>
    public void ClearBenchmarkResults()
    {
        var count = _benchmarkResults.Count;
        _benchmarkResults.Clear();
        _logger.Information("Cleared {Count} benchmark results", count);
    }
    
    /// <summary>
    /// Gets all recorded benchmark results.
    /// </summary>
    /// <returns>An enumerable collection of benchmark results.</returns>
    public IEnumerable<CompressionBenchmarkResult> GetAllBenchmarkResults()
    {
        return _benchmarkResults.AsReadOnly();
    }
}

/// <summary>
/// Represents the result of a model compression benchmark.
/// </summary>
public class CompressionBenchmarkResult
{
    /// <summary>
    /// Gets or sets the name of the model that was compressed.
    /// </summary>
    public string ModelName { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the compression technique that was used.
    /// </summary>
    public string CompressionTechnique { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the original size of the model in bytes.
    /// </summary>
    public long OriginalSizeBytes { get; set; }
    
    /// <summary>
    /// Gets or sets the compressed size of the model in bytes.
    /// </summary>
    public long CompressedSizeBytes { get; set; }
    
    /// <summary>
    /// Gets or sets the compression ratio (original size / compressed size).
    /// </summary>
    public double CompressionRatio { get; set; }
    
    /// <summary>
    /// Gets or sets the impact on model accuracy (negative values indicate accuracy loss).
    /// </summary>
    public double AccuracyImpact { get; set; }
    
    /// <summary>
    /// Gets or sets the inference speedup factor (compressed model speed / original model speed).
    /// </summary>
    public double InferenceSpeedup { get; set; }
    
    /// <summary>
    /// Gets or sets the reduction in memory usage (percentage).
    /// </summary>
    public double MemoryReduction { get; set; }
    
    /// <summary>
    /// Gets or sets the time taken to perform the compression (in milliseconds).
    /// </summary>
    public double CompressionTimeMs { get; set; }

    /// <summary>
    /// Gets or sets hardware information for the benchmark.
    /// </summary>
    public string HardwareInfo { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets additional notes about the benchmark.
    /// </summary>
    public string Notes { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the timestamp when the benchmark was performed.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets additional metrics recorded during the benchmark.
    /// </summary>
    public Dictionary<string, object> AdditionalMetrics { get; set; } = [];
}

internal static class EnumerableExtensions
{
    public static double StdDev(this IEnumerable<double> values)
    {
        var enumerable = values as double[] ?? values.ToArray();
        var avg = enumerable.Average();
        return Math.Sqrt(enumerable.Average(v => Math.Pow(v - avg, 2)));
    }
}