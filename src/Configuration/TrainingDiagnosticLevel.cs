namespace AiDotNet.Configuration;

/// <summary>
/// Verbosity level for training-pipeline diagnostic output (gradient
/// norms, optimizer step traces, tape replay events, etc.).
/// </summary>
/// <remarks>
/// <para>
/// Companion to <see cref="GpuDiagnosticLevel"/>. AiDotNet's training
/// pipeline (TrainWithTape, fused-optimizer fast path, gradient-tape
/// backward) can fail in subtle ways — wrong-direction gradient flow,
/// dropped layer gradients, optimizer skipping parameters. This enum
/// gives consumers fine-grained control over training-side diagnostic
/// output so production code can keep it silent and regression tests
/// can opt in to detailed traces.
/// </para>
/// <para><b>For Beginners:</b> Think of this like log levels in any
/// logging framework: <see cref="Silent"/> is OFF, <see cref="Minimal"/>
/// is "just headline events", <see cref="Verbose"/> is "everything
/// per-batch", <see cref="PerStep"/> is "per-parameter granularity —
/// expensive but exhaustive".
/// </para>
/// </remarks>
public enum TrainingDiagnosticLevel
{
    /// <summary>
    /// No training-pipeline diagnostic output. Default for production code.
    /// </summary>
    Silent = 0,

    /// <summary>
    /// Headline events only: loss value per Train call, optimizer step
    /// failure, gradient explosion/vanish warnings. Cheap to enable
    /// at production scale.
    /// </summary>
    Minimal = 1,

    /// <summary>
    /// Per-batch trace: tape forward/backward boundaries, loss,
    /// aggregate gradient norm, optimizer step result. Useful for
    /// debugging convergence issues without per-parameter overhead.
    /// </summary>
    Verbose = 2,

    /// <summary>
    /// Per-parameter granularity: every parameter tensor's gradient
    /// L2 norm after backward, fused-optimizer fast-path hit/miss,
    /// scheduler ticks. Expensive — only enable for short diagnostic
    /// runs (single Train call or a handful of epochs).
    /// </summary>
    PerStep = 3,
}
