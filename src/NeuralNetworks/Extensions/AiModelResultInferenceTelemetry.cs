using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Extensions.Telemetry;

/// <summary>
/// Unified diagnostics surface for family-specific inference extensions (#1836 excellence
/// goal #4). Reference facades scatter diagnostics per model type (some log via callbacks,
/// some via events, some don't log at all). Here every extension method routes through this
/// helper so callers see one uniform observability contract across radiance-field renders,
/// diffusion generations, transformer decodes, and graph inferences — same session id, same
/// metric names, same shape.
/// </summary>
internal static class AiModelResultInferenceTelemetry
{
    /// <summary>Well-known session id used for all extension-driven inference metrics.</summary>
    public const string InferenceSessionId = "inference";

    /// <summary>
    /// Times the delegate and logs its duration + result-count metric to the model result's
    /// telemetry monitor, if any. When no monitor is attached this is a no-op — extension
    /// methods can wrap every operation without runtime cost for non-observing callers.
    /// </summary>
    public static TResult TimeAndLog<T, TInput, TOutput, TResult>(
        AiModelResult<T, TInput, TOutput> result,
        string operationName,
        Func<TResult> operation,
        int? resultCount = null)
    {
        if (result is null) throw new ArgumentNullException(nameof(result));
        var monitor = result.TrainingMonitor;
        if (monitor is null)
        {
            return operation();
        }
        var sw = Stopwatch.StartNew();
        try
        {
            var value = operation();
            sw.Stop();
            var numOps = MathHelper.GetNumericOperations<T>();
            var metrics = new Dictionary<string, T>
            {
                { $"{operationName}.latency_ms", numOps.FromDouble(sw.Elapsed.TotalMilliseconds) },
            };
            if (resultCount is int rc)
            {
                metrics.Add($"{operationName}.result_count", numOps.FromDouble(rc));
            }
            monitor.LogMetrics(InferenceSessionId, metrics, step: 0);
            return value;
        }
        catch
        {
            sw.Stop();
            var numOps = MathHelper.GetNumericOperations<T>();
            monitor.LogMetric(InferenceSessionId, $"{operationName}.failed_ms", numOps.FromDouble(sw.Elapsed.TotalMilliseconds), step: 0);
            throw;
        }
    }
}
