namespace AiDotNet.Enums;

/// <summary>
/// Policy for the PyTorch GradScaler-style anomaly guard on the Adam
/// optimizer's tape-based <c>Step</c>. The guard scans every gradient
/// element for NaN/Inf and skips the entire step when one is found,
/// preventing permanent poisoning of the <c>m</c>/<c>v</c> moment
/// accumulators.
/// </summary>
public enum AdamAnomalyGuardMode
{
    /// <summary>
    /// Default: scan gradients before each step. Reserved as the
    /// future hook for a numeric-type-aware heuristic (e.g. skip on
    /// fp64 where NaN propagation is much rarer than fp32/fp16).
    /// Currently behaves identically to <see cref="Always"/>.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Always scan gradients before each step. Use when NaN/Inf
    /// gradients are a known failure mode of the training workload
    /// (e.g. fp32 transformer pretraining, GAN training, RL).
    /// </summary>
    Always = 1,

    /// <summary>
    /// Skip the anomaly scan entirely. Use only when upstream
    /// NaN/Inf is impossible (e.g. fully-deterministic fp64
    /// regression tests). Saves O(total-gradient-elements) work
    /// per <c>Step</c>.
    /// </summary>
    Never = 2,
}
