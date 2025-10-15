using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using System;
using System.Collections.Generic;

namespace AiDotNet.Compression.Logging;

/// <summary>
/// Specialized logger for model compression operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This logger tracks the process of model compression, 
/// recording key metrics and events that occur during compression. It helps 
/// with debugging and performance analysis of compression operations.
/// </para>
/// </remarks>
public class CompressionLogger
{
    private readonly ILogging _logger = default!;
    private readonly Dictionary<string, object> _compressionMetrics = new();
    private readonly LoggingOptions _options = default!;
    private readonly string _modelName;
    private readonly string _compressionTechnique;
    
    /// <summary>
    /// Initializes a new instance of the CompressionLogger class.
    /// </summary>
    /// <param name="options">Logging configuration options.</param>
    /// <param name="modelName">Name of the model being compressed.</param>
    /// <param name="compressionTechnique">Compression technique being applied.</param>
    public CompressionLogger(LoggingOptions options, string modelName, string compressionTechnique)
    {
        _options = options;
        _modelName = modelName;
        _compressionTechnique = compressionTechnique;
        
        LoggingFactory.Configure(options);
        _logger = LoggingFactory.GetContextualLogger("ModelName", modelName)
            .ForContext("CompressionTechnique", compressionTechnique);
    }

    /// <summary>
    /// Logs the start of the compression process.
    /// </summary>
    /// <param name="originalModelSize">Size of the original uncompressed model in bytes.</param>
    /// <param name="targetCompressionRatio">Target compression ratio, if specified.</param>
    public void LogCompressionStart(long originalModelSize, double? targetCompressionRatio = null)
    {
        _compressionMetrics["OriginalSizeBytes"] = originalModelSize;
        _compressionMetrics["StartTimestamp"] = DateTime.UtcNow;
        
        if (targetCompressionRatio.HasValue)
        {
            _compressionMetrics["TargetCompressionRatio"] = targetCompressionRatio.Value;
        }
        
        _logger.Information(
            "Starting {Technique} compression for model {Model}. Original size: {Size:N0} bytes, Target ratio: {Ratio}",
            _compressionTechnique,
            _modelName,
            originalModelSize,
            targetCompressionRatio?.ToString("P2") ?? "Not specified");
    }

    /// <summary>
    /// Logs progress during the compression process.
    /// </summary>
    /// <param name="stage">Current compression stage.</param>
    /// <param name="progressPercentage">Percentage of completion.</param>
    /// <param name="message">Optional additional message.</param>
    public void LogCompressionProgress(string stage, double progressPercentage, string? message = null)
    {
        _logger.Debug(
            "Compression progress: {Stage} - {Progress:P2} complete. {Message}", 
            stage, 
            progressPercentage,
            message ?? string.Empty);
    }

    /// <summary>
    /// Logs a specific compression metric.
    /// </summary>
    /// <param name="metricName">Name of the metric.</param>
    /// <param name="value">Value of the metric.</param>
    public void LogCompressionMetric(string metricName, object value)
    {
        _compressionMetrics[metricName] = value;
        _logger.Debug("Compression metric: {MetricName} = {MetricValue}", metricName, value);
    }

    /// <summary>
    /// Logs a warning during the compression process.
    /// </summary>
    /// <param name="message">Warning message.</param>
    /// <param name="args">Additional format arguments.</param>
    public void LogWarning(string message, params object[] args)
    {
        _logger.Warning(message, args);
    }

    /// <summary>
    /// Logs an error that occurred during compression.
    /// </summary>
    /// <param name="exception">The exception that was thrown.</param>
    /// <param name="message">Error message.</param>
    /// <param name="args">Additional format arguments.</param>
    public void LogError(Exception exception, string message, params object[] args)
    {
        _logger.Error(exception, message, args);
    }

    /// <summary>
    /// Logs an error message.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="args">Additional format arguments.</param>
    public void Error(string message, params object[] args)
    {
        _logger.Error(message, args);
    }

    /// <summary>
    /// Logs the completion of the compression process.
    /// </summary>
    /// <param name="compressedModelSize">Size of the compressed model in bytes.</param>
    /// <param name="compressionRatio">Actual compression ratio achieved.</param>
    /// <param name="accuracyImpact">Impact on model accuracy, if measured.</param>
    /// <param name="speedupFactor">Inference speedup factor, if measured.</param>
    public void LogCompressionComplete(
        long compressedModelSize, 
        double compressionRatio, 
        double? accuracyImpact = null, 
        double? speedupFactor = null)
    {
        var endTime = DateTime.UtcNow;
        var startTime = (DateTime)_compressionMetrics["StartTimestamp"];
        var duration = endTime - startTime;
        
        _compressionMetrics["CompressedSizeBytes"] = compressedModelSize;
        _compressionMetrics["CompressionRatio"] = compressionRatio;
        _compressionMetrics["DurationMs"] = duration.TotalMilliseconds;
        
        if (accuracyImpact.HasValue)
        {
            _compressionMetrics["AccuracyImpact"] = accuracyImpact.Value;
        }
        
        if (speedupFactor.HasValue)
        {
            _compressionMetrics["SpeedupFactor"] = speedupFactor.Value;
        }

        _logger.Information(
            "Completed {Technique} compression for model {Model} in {Duration:g}. " +
            "Original: {OriginalSize:N0} bytes, Compressed: {CompressedSize:N0} bytes, " +
            "Ratio: {Ratio:P2}, Accuracy impact: {Accuracy}, Speedup: {Speedup:F2}x",
            _compressionTechnique,
            _modelName,
            duration,
            _compressionMetrics["OriginalSizeBytes"],
            compressedModelSize,
            compressionRatio,
            accuracyImpact?.ToString("P2") ?? "Not measured",
            speedupFactor ?? 0);
    }

    /// <summary>
    /// Gets a dictionary containing all recorded compression metrics.
    /// </summary>
    /// <returns>A dictionary of compression metrics.</returns>
    public IReadOnlyDictionary<string, object> GetCompressionMetrics()
    {
        return _compressionMetrics;
    }
    
    /// <summary>
    /// Gets a logger instance for a specific compression phase or component.
    /// </summary>
    /// <param name="phase">The compression phase or component name.</param>
    /// <returns>A specialized logger for the specified phase.</returns>
    public CompressionLogger ForPhase(string phase)
    {
        var phaseLogger = new CompressionLogger(_options, _modelName, _compressionTechnique);
        phaseLogger._logger.ForContext("Phase", phase);
        
        // Copy the existing metrics 
        foreach (var metric in _compressionMetrics)
        {
            phaseLogger._compressionMetrics[metric.Key] = metric.Value;
        }
        
        return phaseLogger;
    }
}