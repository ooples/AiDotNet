namespace AiDotNet.Configuration;

/// <summary>
/// Process-global control for GPU backend diagnostic output visibility.
/// Exposes three orthogonal knobs matching
/// github.com/ooples/AiDotNet#1122's requested-change checklist:
/// environment variable, static configuration, and ILogger / custom sink.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet's GPU backends (OpenCL, HIP, CUDA) emit status messages during
/// device discovery, kernel compilation, and availability checks.
/// Historically these were always written to <see cref="System.Console.WriteLine"/>,
/// producing ~40 lines of output on every <c>AiModelBuilder.BuildAsync()</c>.
/// This class provides the AiDotNet-side facade for controlling that output.
/// </para>
/// <para><b>Three orthogonal controls (all work simultaneously):</b></para>
/// <list type="number">
/// <item><b>Environment variable</b> — <c>AIDOTNET_GPU_VERBOSE=1</c>,
/// <c>=true</c>, or <c>=verbose</c> at process start enables
/// <see cref="GpuDiagnosticLevel.Verbose"/>. <c>=minimal</c> sets
/// <see cref="GpuDiagnosticLevel.Minimal"/>. Anything else (including
/// unset) leaves the level at <see cref="GpuDiagnosticLevel.Silent"/>.</item>
/// <item><b>Static configuration</b> — set <see cref="Level"/> to
/// <see cref="GpuDiagnosticLevel.Silent"/>, <see cref="GpuDiagnosticLevel.Minimal"/>,
/// or <see cref="GpuDiagnosticLevel.Verbose"/>. Assignment takes effect
/// immediately for subsequent diagnostic emissions.</item>
/// <item><b>Sink / ILogger</b> — assign <see cref="Sink"/> to route
/// diagnostic messages through your logging framework instead of
/// <see cref="System.Console"/>. See
/// <see cref="GpuDiagnosticsLoggerExtensions.ToSink(Microsoft.Extensions.Logging.ILogger)"/>
/// for ILogger integration.</item>
/// </list>
/// </remarks>
/// <example>
/// <code>
/// // Option A: Environment variable — set before starting the process.
/// //   AIDOTNET_GPU_VERBOSE=1
///
/// // Option B: Static configuration
/// AiDotNet.Configuration.GpuDiagnosticsConfig.Level = GpuDiagnosticLevel.Silent;
///
/// // Option C: Custom sink (route to any logging framework).
/// AiDotNet.Configuration.GpuDiagnosticsConfig.Sink =
///     (level, msg) => myLogger.LogInformation("[gpu:{Level}] {Msg}", level, msg);
///
/// // Or via ILogger extension
/// AiDotNet.Configuration.GpuDiagnosticsConfig.Sink = myLogger.ToSink();
/// </code>
/// </example>
public static class GpuDiagnosticsConfig
{
    private static GpuDiagnosticLevel _level = InitLevelFromEnvironment();
    private static GpuDiagnosticSink? _sink;

    /// <summary>
    /// Current verbosity level. Reads and writes are synchronized at the
    /// primitive-assignment level — a concurrent producer/consumer race
    /// produces only one stale-read message, never corruption.
    /// </summary>
    /// <remarks>
    /// Setting <see cref="Level"/> synchronously forwards to
    /// <see cref="AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput"/>
    /// so the underlying Tensors package honors the level on its next
    /// diagnostic emission. <see cref="GpuDiagnosticLevel.Silent"/> and
    /// <see cref="GpuDiagnosticLevel.Minimal"/> both map to <c>false</c>
    /// on the current Tensors v0.38.0 (which supports bool-level only);
    /// <see cref="GpuDiagnosticLevel.Verbose"/> maps to <c>true</c>.
    /// </remarks>
    public static GpuDiagnosticLevel Level
    {
        get => _level;
        set
        {
            _level = value;
            // Forward to Tensors layer. Silent/Minimal both suppress because
            // Tensors v0.38.0 only has a bool toggle — it doesn't yet support
            // per-message level tagging. Minimal-specific filtering becomes
            // meaningful once the Tensors package adds sink-based routing.
            AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput =
                value == GpuDiagnosticLevel.Verbose;
        }
    }

    /// <summary>
    /// Legacy bool flag. Kept for source compatibility with callers written
    /// against the first iteration of this API.
    /// <c>true</c> ≡ <see cref="GpuDiagnosticLevel.Verbose"/>;
    /// <c>false</c> ≡ <see cref="GpuDiagnosticLevel.Silent"/>.
    /// </summary>
    /// <remarks>
    /// Setting <see cref="Verbose"/> to <c>true</c> is equivalent to
    /// <c>Level = GpuDiagnosticLevel.Verbose</c>. Setting to <c>false</c>
    /// collapses both Silent and Minimal onto <see cref="GpuDiagnosticLevel.Silent"/> —
    /// prefer the <see cref="Level"/> property when the distinction matters.
    /// </remarks>
    public static bool Verbose
    {
        get => _level == GpuDiagnosticLevel.Verbose;
        set => Level = value ? GpuDiagnosticLevel.Verbose : GpuDiagnosticLevel.Silent;
    }

    /// <summary>
    /// Optional sink that receives diagnostic messages when set. When
    /// <c>null</c>, diagnostic output goes directly to
    /// <see cref="System.Console.WriteLine"/> at the Tensors layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Forward-compatibility:</b> the sink is stored immediately, but the
    /// Tensors v0.38.0 backend writes to Console directly. On Tensors
    /// v0.39+ (when sink routing lands), emissions will be delivered to
    /// the sink WITH level tagging. For now, setting a sink AND
    /// <see cref="Level"/> = <c>Verbose</c> produces duplicate output
    /// (Console AND future-sink invocation); combine the sink with
    /// <see cref="Level"/> = <c>Silent</c> to route only through the sink.
    /// </para>
    /// <para>
    /// For <see cref="Microsoft.Extensions.Logging.ILogger"/> integration,
    /// use
    /// <see cref="GpuDiagnosticsLoggerExtensions.ToSink(Microsoft.Extensions.Logging.ILogger)"/>.
    /// </para>
    /// </remarks>
    public static GpuDiagnosticSink? Sink
    {
        get => _sink;
        set => _sink = value;
    }

    /// <summary>
    /// Emits a diagnostic message, respecting the current <see cref="Level"/>
    /// and routing through <see cref="Sink"/> if set (else Console).
    /// Callable from AiDotNet-side diagnostic code that wants to participate
    /// in the same level/sink pipeline the Tensors-side ultimately will.
    /// </summary>
    /// <param name="level">
    /// The severity of this message. Emission is suppressed when
    /// <paramref name="level"/> is less severe than <see cref="Level"/>
    /// (numerically lower enum value). Levels ranked Silent &lt; Minimal &lt; Verbose.
    /// </param>
    /// <param name="message">The diagnostic message text.</param>
    public static void Emit(GpuDiagnosticLevel level, string message)
    {
        // Suppress when the active Level is more restrictive than the message's level.
        // Silent = 0 suppresses everything (including Minimal messages).
        // Minimal = 1 permits Minimal + Verbose? No — Minimal permits Minimal-severity
        //   messages only. Verbose messages need level=Verbose.
        // So emit if current level >= message level in severity (numerically >=).
        if (_level < level) return;
        var sink = _sink;
        if (sink is not null)
        {
            sink(level, message);
            return;
        }
        System.Console.WriteLine(message);
    }

    private static GpuDiagnosticLevel InitLevelFromEnvironment()
    {
        var val = System.Environment.GetEnvironmentVariable("AIDOTNET_GPU_VERBOSE");
        if (string.IsNullOrWhiteSpace(val)) return GpuDiagnosticLevel.Silent;
        if (val!.Equals("1", System.StringComparison.OrdinalIgnoreCase) ||
            val.Equals("true", System.StringComparison.OrdinalIgnoreCase) ||
            val.Equals("yes", System.StringComparison.OrdinalIgnoreCase) ||
            val.Equals("on", System.StringComparison.OrdinalIgnoreCase) ||
            val.Equals("verbose", System.StringComparison.OrdinalIgnoreCase))
            return GpuDiagnosticLevel.Verbose;
        if (val.Equals("minimal", System.StringComparison.OrdinalIgnoreCase))
            return GpuDiagnosticLevel.Minimal;
        return GpuDiagnosticLevel.Silent;
    }
}
