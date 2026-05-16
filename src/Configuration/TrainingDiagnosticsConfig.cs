using System;

namespace AiDotNet.Configuration;

/// <summary>
/// Process-global control for training-pipeline diagnostic output
/// (gradient norms, optimizer step traces, tape events). Mirrors the
/// three-orthogonal-knobs design of <see cref="GpuDiagnosticsConfig"/>:
/// environment variable, static configuration, and custom sink.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet's training pipeline (<c>NeuralNetworkBase.TrainWithTape</c>)
/// can fail in subtle ways — wrong-direction gradient flow, dropped
/// layer gradients, optimizer skipping parameters, fused-path bailouts.
/// These bugs are hard to diagnose without per-parameter gradient
/// visibility (see github.com/ooples/AiDotNet#1328 for an example where
/// the model trained but only the bias of the final dense head got
/// useful gradients). This class provides production-grade hooks so
/// regression suites and end-user debugging can introspect training
/// without modifying the library.
/// </para>
/// <para><b>Three orthogonal controls</b> (all work simultaneously):</para>
/// <list type="number">
/// <item><b>Environment variable</b> — <c>AIDOTNET_TRAINING_DIAGNOSTICS</c>
///   set to <c>minimal</c>, <c>verbose</c>, or <c>perstep</c> at process
///   start initializes <see cref="Level"/>. Unset leaves it at
///   <see cref="TrainingDiagnosticLevel.Silent"/>.</item>
/// <item><b>Static configuration</b> — set <see cref="Level"/> directly
///   at runtime. Assignment takes effect for the next training step.</item>
/// <item><b>Custom sink</b> — assign <see cref="Sink"/> to route events
///   through your logging framework. When <c>null</c>, events at or below
///   the active level are routed to <see cref="System.Diagnostics.Trace"/>.</item>
/// </list>
/// </remarks>
/// <example>
/// <code>
/// // Option A: environment variable (set before process start)
/// //   AIDOTNET_TRAINING_DIAGNOSTICS=perstep
///
/// // Option B: static config
/// AiDotNet.Configuration.TrainingDiagnosticsConfig.Level =
///     TrainingDiagnosticLevel.PerStep;
///
/// // Option C: custom sink that collects GradientNormEvents for assertions
/// var grads = new System.Collections.Generic.List&lt;GradientNormEvent&gt;();
/// AiDotNet.Configuration.TrainingDiagnosticsConfig.Sink = evt =&gt; {
///     if (evt is GradientNormEvent g) grads.Add(g);
/// };
/// AiDotNet.Configuration.TrainingDiagnosticsConfig.Level =
///     TrainingDiagnosticLevel.PerStep;
/// model.Train(input, target);
/// AiDotNet.Configuration.TrainingDiagnosticsConfig.Level =
///     TrainingDiagnosticLevel.Silent;
/// // grads now has one record per trainable parameter for that step.
/// </code>
/// </example>
public static class TrainingDiagnosticsConfig
{
    private static TrainingDiagnosticLevel _level = InitLevelFromEnvironment();
    private static TrainingDiagnosticSink? _sink;
    private static int _stepCounter;

    /// <summary>
    /// Current verbosity level. Reads and writes are not synchronized
    /// beyond reference-assignment atomicity — a concurrent producer /
    /// consumer race can produce one stale read but never corruption.
    /// </summary>
    public static TrainingDiagnosticLevel Level
    {
        get => _level;
        set => _level = value;
    }

    /// <summary>
    /// Optional sink that receives diagnostic events when set. When
    /// <c>null</c>, events at or below <see cref="Level"/> are routed
    /// to <see cref="System.Diagnostics.Trace"/>.
    /// </summary>
    /// <remarks>
    /// Sinks are invoked synchronously from inside the training hot
    /// loop — keep work cheap. For heavy logging, queue the event and
    /// process asynchronously.
    /// </remarks>
    public static TrainingDiagnosticSink? Sink
    {
        get => _sink;
        set => _sink = value;
    }

    /// <summary>
    /// Convenience boolean for the most common case ("turn on per-step
    /// gradient diagnostics for one test"). <c>true</c> is equivalent to
    /// <see cref="TrainingDiagnosticLevel.PerStep"/>; <c>false</c> resets
    /// to <see cref="TrainingDiagnosticLevel.Silent"/>.
    /// </summary>
    public static bool PerStepEnabled
    {
        get => _level >= TrainingDiagnosticLevel.PerStep;
        set => _level = value ? TrainingDiagnosticLevel.PerStep : TrainingDiagnosticLevel.Silent;
    }

    /// <summary>
    /// Monotonically incrementing step counter, advanced once per call
    /// to <c>TrainWithTape</c>. Used as the StepIndex on emitted events
    /// so consumers can correlate per-parameter records into a single
    /// training step. Test-side code can <see cref="ResetStepCounter"/>
    /// to start a clean run.
    /// </summary>
    public static int CurrentStepIndex => _stepCounter;

    public static void ResetStepCounter() =>
        System.Threading.Interlocked.Exchange(ref _stepCounter, 0);

    /// <summary>Internal — advances and returns the new step index.</summary>
    internal static int AdvanceStep() => System.Threading.Interlocked.Increment(ref _stepCounter);

    /// <summary>
    /// Emits a structured diagnostic event respecting the current
    /// <see cref="Level"/>. Routes through <see cref="Sink"/> if set,
    /// else to <see cref="System.Diagnostics.Trace"/>.
    /// </summary>
    /// <param name="evt">The event to emit.</param>
    /// <remarks>
    /// A throwing sink must not break training, so sink exceptions are
    /// caught here. The exception is forwarded to
    /// <see cref="System.Diagnostics.Trace.TraceError(string)"/> with the
    /// event type, level, and full <c>ex.ToString()</c> so a sink
    /// regression is observable via standard .NET trace listeners rather
    /// than silently dropped.
    /// </remarks>
    public static void Emit(TrainingDiagnosticEvent evt)
    {
        if (evt is null) return;
        if (_level < evt.Level) return;
        var sink = _sink;
        if (sink is not null)
        {
            try { sink(evt); }
            catch (Exception ex)
            {
                // Sink failure is reported but never propagates — training
                // must not be broken by instrumentation.
                // System.Diagnostics.Trace.TraceError does NOT internally
                // catch exceptions thrown by registered TraceListeners
                // (e.g., a network logger that times out, a file listener
                // that hits ENOSPC). Wrap the fallback in its own
                // try/catch so a broken listener can't abort training
                // either.
                try
                {
                    System.Diagnostics.Trace.TraceError(
                        "TrainingDiagnostics sink failed for {0} at level {1}: {2}",
                        evt.GetType().Name, evt.Level, ex);
                }
                catch
                {
                    // Never let fallback diagnostics abort training.
                }
            }
            return;
        }
        // Same listener-safety guard as above — the default Trace path
        // must not propagate listener exceptions.
        try
        {
            System.Diagnostics.Trace.WriteLine(evt.ToString());
        }
        catch
        {
            // Never let fallback diagnostics abort training.
        }
    }

    /// <summary>
    /// Convenience helper for emitting a free-form message at a specific
    /// level. Equivalent to
    /// <c>Emit(new TrainingMessageEvent(level, message))</c>.
    /// </summary>
    public static void EmitMessage(TrainingDiagnosticLevel level, string message)
        => Emit(new TrainingMessageEvent(level, message));

    private static TrainingDiagnosticLevel InitLevelFromEnvironment()
    {
        var val = Environment.GetEnvironmentVariable("AIDOTNET_TRAINING_DIAGNOSTICS");
        if (string.IsNullOrWhiteSpace(val)) return TrainingDiagnosticLevel.Silent;
        // Trim leading/trailing whitespace so values from container env
        // templating (e.g. " perstep\n") still match.
        val = val!.Trim();
        if (val.Equals("minimal", StringComparison.OrdinalIgnoreCase))
            return TrainingDiagnosticLevel.Minimal;
        if (val.Equals("verbose", StringComparison.OrdinalIgnoreCase))
            return TrainingDiagnosticLevel.Verbose;
        if (val.Equals("perstep", StringComparison.OrdinalIgnoreCase) ||
            val.Equals("per-step", StringComparison.OrdinalIgnoreCase) ||
            val.Equals("per_step", StringComparison.OrdinalIgnoreCase))
            return TrainingDiagnosticLevel.PerStep;
        if (val.Equals("1", StringComparison.OrdinalIgnoreCase) ||
            val.Equals("true", StringComparison.OrdinalIgnoreCase) ||
            val.Equals("on", StringComparison.OrdinalIgnoreCase))
            return TrainingDiagnosticLevel.Verbose;
        return TrainingDiagnosticLevel.Silent;
    }
}
