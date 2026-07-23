namespace AiDotNet.Configuration;

/// <summary>
/// Options for controlling GPU backend diagnostic output visibility.
/// Addresses github.com/ooples/AiDotNet#1122 — all three requested
/// controls (environment variable, static configuration, ILogger /
/// custom sink) are reachable through this options class or the
/// underlying <see cref="GpuDiagnosticsConfig"/> static facade.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet's GPU backends (OpenCL, HIP, CUDA) emit status messages during
/// device discovery, kernel compilation, and availability checks. This
/// options class lets applications configure the verbosity and routing
/// of that output via the fluent
/// <see cref="AiDotNet.Interfaces.IAiModelBuilder{T, TInput, TOutput}.ConfigureGpuDiagnostics(GpuDiagnosticsOptions)"/>
/// builder method.
/// </para>
/// <para>
/// All properties are nullable — <c>null</c> means "don't change the
/// current setting", so passing an empty options instance is a no-op.
/// This matches the AiDotNet facade pattern
/// (<c>TelemetryConfig</c> / <c>ProfilingConfig</c>).
/// </para>
/// <para><b>For Beginners:</b> If your AI application is printing lots of
/// <c>[OpenClBackend] Compiling kernels...</c> messages, pass
/// <c>new GpuDiagnosticsOptions { Level = GpuDiagnosticLevel.Silent }</c>
/// to the builder's <c>ConfigureGpuDiagnostics</c> method. If you want
/// them routed through your logger instead, set <see cref="Sink"/>.</para>
/// </remarks>
/// <example>
/// <code>
/// // Silence all GPU diagnostics.
/// builder.ConfigureGpuDiagnostics(new() { Level = GpuDiagnosticLevel.Silent });
///
/// // Verbose for troubleshooting.
/// builder.ConfigureGpuDiagnostics(new() { Level = GpuDiagnosticLevel.Verbose });
///
/// // Route through an ILogger.
/// builder.ConfigureGpuDiagnostics(new()
/// {
///     Level = GpuDiagnosticLevel.Minimal,
///     Sink = logger.ToSink()
/// });
/// </code>
/// </example>
public class GpuDiagnosticsOptions
{
    /// <summary>
    /// Verbosity level. <c>null</c> preserves the current
    /// <see cref="GpuDiagnosticsConfig.Level"/> (set by env var or prior
    /// programmatic assignment). Explicit values override.
    /// </summary>
    public GpuDiagnosticLevel? Level { get; set; }

    /// <summary>
    /// Optional sink that receives diagnostic messages. <c>null</c>
    /// preserves the current <see cref="GpuDiagnosticsConfig.Sink"/>.
    /// Pass a non-null delegate to register a custom sink (e.g.
    /// <c>logger.ToSink()</c>); setting to <c>null</c> does NOT clear an
    /// already-registered sink (preservation semantics). To UNREGISTER
    /// a sink, set <see cref="GpuDiagnosticsConfig.Sink"/> directly.
    /// </summary>
    public GpuDiagnosticSink? Sink { get; set; }

    /// <summary>
    /// Legacy bool-level flag. Kept for source compatibility with
    /// callers written against the first iteration of this API.
    /// <c>true</c> ≡ <see cref="GpuDiagnosticLevel.Verbose"/>.
    /// <c>false</c> ≡ <see cref="GpuDiagnosticLevel.Silent"/>.
    /// <c>null</c> preserves.
    /// </summary>
    /// <remarks>
    /// When BOTH <see cref="Verbose"/> and <see cref="Level"/> are set,
    /// <see cref="Level"/> wins — it's the richer API. Applications
    /// should prefer <see cref="Level"/> in new code.
    /// </remarks>
    public bool? Verbose { get; set; }
}
