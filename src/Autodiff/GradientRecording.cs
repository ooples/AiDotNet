using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Autodiff;

/// <summary>
/// Whether the current forward is being traced for a compiled training plan.
/// </summary>
/// <remarks>
/// A compiled training plan traces the forward with no <c>GradientTape</c> active, then differentiates the graph it
/// recorded. A layer that picks an in-place or pooled-buffer fast path whenever no tape is active therefore takes it
/// during that trace too, and the plan cannot differentiate through it: every weight upstream becomes a constant and
/// receives no gradient. Such fast paths must also stay off while this is true. An inference trace is not affected.
/// </remarks>
internal static class GradientRecording
{
    /// <summary>True while the forward is being traced for a compiled training plan.</summary>
    public static bool IsTrainingTrace => GraphMode.IsActive && !GraphMode.IsInferenceTrace;
}