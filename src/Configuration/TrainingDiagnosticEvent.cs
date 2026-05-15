using System;
using AiDotNet.Interfaces;

namespace AiDotNet.Configuration;

/// <summary>
/// Base type for structured training-pipeline diagnostic events.
/// Sinks can dispatch on the runtime type to render or filter
/// per-event-type without parsing string payloads.
/// </summary>
public abstract record TrainingDiagnosticEvent(TrainingDiagnosticLevel Level)
{
    /// <summary>Default text rendering — sinks that want raw text can call this.</summary>
    public override string ToString() => $"[{Level}] {GetType().Name}";
}

/// <summary>
/// Emitted once per call to <c>TrainWithTape</c> with the scalar loss
/// value returned by the loss function for that batch. Use to track
/// loss trajectory across epochs without hand-instrumenting the loop.
/// </summary>
public sealed record TrainingLossEvent(
    int StepIndex,
    double LossValue,
    int OutputRank,
    long OutputLength)
    : TrainingDiagnosticEvent(TrainingDiagnosticLevel.Minimal)
{
    public override string ToString() =>
        $"[Minimal] TrainingLoss step={StepIndex} loss={LossValue:E4} outputRank={OutputRank} outputLen={OutputLength}";
}

/// <summary>
/// Emitted per parameter tensor after <c>tape.ComputeGradients</c>
/// returns, before the optimizer step runs. Captures the L2 norm of the
/// gradient flowing back into that tensor. The fingerprint of
/// gradient-flow bugs (issue #1328) is: head-side params show non-zero
/// norm but embedding / attention QKV / output-projection norms are
/// zero or near-zero.
/// </summary>
public sealed record GradientNormEvent : TrainingDiagnosticEvent
{
    /// <summary>Sequence number of the training step that emitted this event.</summary>
    public int StepIndex { get; }
    /// <summary>Position of this parameter in the network's enumerated trainable-tensor list.</summary>
    public int ParamIndex { get; }
    /// <summary>Defensive snapshot of the parameter tensor's shape at emission time.</summary>
    public int[] ParamShape { get; }
    /// <summary>Total scalar element count of the parameter tensor.</summary>
    public long ParamLength { get; }
    /// <summary>True when the gradient tape produced a gradient for this parameter.</summary>
    public bool HasGradient { get; }
    /// <summary>L2 norm of the gradient at emission time; 0 when <see cref="HasGradient"/> is false.</summary>
    public double GradientL2Norm { get; }
    /// <summary>Type-safe categorization of the owning layer.</summary>
    public LayerCategory LayerCategory { get; }
    /// <summary>Concrete layer-class name for diagnostic readability.</summary>
    public string LayerTypeName { get; }

    public GradientNormEvent(
        int StepIndex,
        int ParamIndex,
        int[] ParamShape,
        long ParamLength,
        bool HasGradient,
        double GradientL2Norm,
        LayerCategory LayerCategory,
        string LayerTypeName)
        : base(TrainingDiagnosticLevel.PerStep)
    {
        this.StepIndex = StepIndex;
        this.ParamIndex = ParamIndex;
        // Defensive copy so the emitted event is an immutable snapshot —
        // mutations of the source array after emission cannot corrupt
        // diagnostics consumers reading the event later.
        this.ParamShape = ParamShape is null
            ? System.Array.Empty<int>()
            : (int[])ParamShape.Clone();
        this.ParamLength = ParamLength;
        this.HasGradient = HasGradient;
        this.GradientL2Norm = GradientL2Norm;
        this.LayerCategory = LayerCategory;
        this.LayerTypeName = LayerTypeName;
    }

    public override string ToString()
    {
        string sh = ParamShape.Length == 0 ? "[]" : "[" + string.Join(",", ParamShape) + "]";
        return HasGradient
            ? $"[PerStep] GradientNorm step={StepIndex} p{ParamIndex:D3} cat={LayerCategory} type={LayerTypeName} shape={sh} len={ParamLength} ||grad||={GradientL2Norm:E4}"
            : $"[PerStep] GradientNorm step={StepIndex} p{ParamIndex:D3} cat={LayerCategory} type={LayerTypeName} shape={sh} len={ParamLength} ||grad||=NO_GRAD";
    }
}

/// <summary>
/// Emitted when <c>TrainWithTape</c> enters or skips the fused-compiled
/// fast path. Hit means forward + backward + optimizer step ran in a
/// single compiled kernel; miss falls back to the eager tape walk.
/// </summary>
public sealed record FusedOptimizerPathEvent(
    int StepIndex,
    bool Hit,
    string? Reason)
    : TrainingDiagnosticEvent(TrainingDiagnosticLevel.PerStep)
{
    public override string ToString() =>
        Hit
            ? $"[PerStep] FusedOptimizer step={StepIndex} HIT"
            : $"[PerStep] FusedOptimizer step={StepIndex} MISS reason={Reason ?? "(unspecified)"}";
}

/// <summary>
/// Free-form text event for callers that just want to emit a typed
/// log line at a specific level. Prefer the structured events above
/// when the data is repeating-shape-friendly.
/// </summary>
public sealed record TrainingMessageEvent(
    TrainingDiagnosticLevel MsgLevel,
    string Message)
    : TrainingDiagnosticEvent(MsgLevel)
{
    public override string ToString() => $"[{MsgLevel}] {Message}";
}
