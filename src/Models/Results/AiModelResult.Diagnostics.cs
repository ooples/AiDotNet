using AiDotNet.Diagnostics;

namespace AiDotNet.Models.Results;

/// <summary>
/// Diagnostics surface on <see cref="AiModelResult{T, TInput, TOutput}"/> — exposes the
/// acceleration environment snapshot captured at build time when the builder opts in via
/// <c>ReportAccelerationStatus()</c>.
/// </summary>
public partial class AiModelResult<T, TInput, TOutput>
{
    /// <summary>
    /// Snapshot of the SIMD, GPU, and native-BLAS acceleration state captured when this
    /// model was built. <c>null</c> if the builder did not call <c>ReportAccelerationStatus</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Useful for production observability, CI assertions, and diagnosing why a model is
    /// slower than expected on a given host. Fields include CPU SIMD level
    /// (<see cref="AccelerationSnapshot.BestSimdSet"/>), GPU backends detected
    /// (<see cref="AccelerationSnapshot.HasCuda"/>, etc.), and native BLAS availability
    /// (<see cref="AccelerationSnapshot.HasOpenBlas"/>, etc.).
    /// </para>
    /// </remarks>
    public AccelerationSnapshot? AccelerationSnapshot { get; internal set; }

    /// <summary>
    /// Per-tensor-op performance profile captured when the builder opted in via
    /// <c>EnableTensorsOpProfiling()</c>. Null otherwise. Complements the
    /// higher-level <c>ProfilingReport</c> (from <c>ConfigureProfiling</c>) by
    /// surfacing Tensors-package kernel timings, not just AiDotNet workflow timings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// PyTorch-parity equivalent: low-level <c>torch.profiler.profile</c> CUDA/CPU op
    /// breakdown. Operations are sorted by total time descending; use
    /// <see cref="TensorsOperationProfile.FormatSummary"/> for a one-line-per-op table.
    /// </para>
    /// </remarks>
    public TensorsOperationProfile? TensorsOperationProfile { get; internal set; }
}
