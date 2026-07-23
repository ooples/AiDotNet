using System;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Configuration;

/// <summary>
/// Extension methods that adapt
/// <see cref="Microsoft.Extensions.Logging.ILogger"/> into a
/// <see cref="GpuDiagnosticSink"/>. Addresses Option C of
/// github.com/ooples/AiDotNet#1122:
/// "Use ILogger instead of Console.WriteLine — Then applications control
/// output through their logging framework."
/// </summary>
public static class GpuDiagnosticsLoggerExtensions
{
    /// <summary>
    /// Wraps an <see cref="ILogger"/> as a
    /// <see cref="GpuDiagnosticSink"/>, mapping
    /// <see cref="GpuDiagnosticLevel"/> onto <see cref="Microsoft.Extensions.Logging.LogLevel"/>:
    /// <see cref="GpuDiagnosticLevel.Silent"/> → no emission (sink
    /// never fires when level is Silent because the level-gate
    /// upstream drops the message),
    /// <see cref="GpuDiagnosticLevel.Minimal"/> → <see cref="Microsoft.Extensions.Logging.LogLevel.Information"/>,
    /// <see cref="GpuDiagnosticLevel.Verbose"/> → <see cref="Microsoft.Extensions.Logging.LogLevel.Debug"/>.
    /// </summary>
    /// <param name="logger">
    /// The logger to route GPU diagnostics through. Must not be null.
    /// </param>
    /// <returns>
    /// A sink delegate that can be assigned to
    /// <see cref="GpuDiagnosticsConfig.Sink"/>.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="logger"/> is null.
    /// </exception>
    /// <example>
    /// <code>
    /// services.AddLogging();
    /// var logger = serviceProvider.GetRequiredService&lt;ILogger&lt;MyApp&gt;&gt;();
    /// AiDotNet.Configuration.GpuDiagnosticsConfig.Sink = logger.ToSink();
    /// </code>
    /// </example>
    /// <remarks>
    /// The mapping rationale:
    /// <list type="bullet">
    /// <item><see cref="GpuDiagnosticLevel.Minimal"/> is surfaced at
    /// <see cref="Microsoft.Extensions.Logging.LogLevel.Information"/> because "GPU initialized: NVIDIA RTX"
    /// is useful operational info — default log levels include Information,
    /// so Minimal messages survive most filters.</item>
    /// <item><see cref="GpuDiagnosticLevel.Verbose"/> is surfaced at
    /// <see cref="Microsoft.Extensions.Logging.LogLevel.Debug"/> because kernel-compile progress /
    /// GEMM-tuning dumps are implementation-detail chatter — default
    /// log levels filter Debug, so Verbose messages only surface when
    /// the application explicitly enables Debug logging.</item>
    /// </list>
    /// </remarks>
    public static GpuDiagnosticSink ToSink(this ILogger logger)
    {
        if (logger is null) throw new ArgumentNullException(nameof(logger));
        return (level, message) =>
        {
            Microsoft.Extensions.Logging.LogLevel logLevel = level switch
            {
                GpuDiagnosticLevel.Verbose => Microsoft.Extensions.Logging.LogLevel.Debug,
                GpuDiagnosticLevel.Minimal => Microsoft.Extensions.Logging.LogLevel.Information,
                _ => Microsoft.Extensions.Logging.LogLevel.Trace, // Silent shouldn't actually reach the sink
            };
            logger.Log(logLevel, message);
        };
    }
}
