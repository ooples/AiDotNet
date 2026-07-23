namespace AiDotNet.Configuration;

/// <summary>
/// Delegate that receives training-pipeline diagnostic events in lieu of
/// the default <see cref="System.Diagnostics.Trace.WriteLine(string)"/> path.
/// Applications register a sink to route training diagnostics through
/// their logging framework of choice (Spectre.Console, Serilog,
/// Microsoft.Extensions.Logging, structured logs, OpenTelemetry, etc.).
/// </summary>
/// <param name="evt">
/// The structured diagnostic event. Switch on the runtime type
/// (<c>GradientNormEvent</c>, <c>TrainingLossEvent</c>,
/// <c>FusedOptimizerPathEvent</c>, <c>TrainingMessageEvent</c>) to
/// render or filter per-event-type. Each event carries its own
/// <see cref="TrainingDiagnosticEvent.Level"/> so the sink can match
/// the GpuDiagnostics filtering pattern even without consulting
/// <see cref="TrainingDiagnosticsConfig.Level"/>.
/// </param>
/// <remarks>
/// Sinks are invoked synchronously from inside the training hot loop —
/// keep work cheap. For heavy logging, queue the event and process
/// asynchronously.
///
/// <para>Sink exceptions are caught by
/// <see cref="TrainingDiagnosticsConfig.Emit"/> and reported via
/// <see cref="System.Diagnostics.Trace.TraceError(string)"/> so a
/// throwing sink cannot break training. There is no opt-in to
/// rethrow / fail-fast semantics today — if a caller needs that, file
/// a feature request. Sinks should still avoid throwing in hot-path
/// instrumentation: the Trace.TraceError fallback is synchronous and
/// runs in the training step's critical path.</para>
/// </remarks>
public delegate void TrainingDiagnosticSink(TrainingDiagnosticEvent evt);
