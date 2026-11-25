using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using PrecisionMode = AiDotNet.Tensors.Engines.Gpu.PrecisionMode;

namespace AiDotNet.MixedPrecision;

/// <summary>
/// Mixed precision on a GPU engine, run as PyTorch's <c>torch.autocast</c> runs it: the FP32 master weights stay as
/// they are, and every matrix product of the step's forward and backward casts its operands to FP16 or BF16 and
/// accumulates in FP32 (<see cref="DirectGpuTensorEngine.EnterMatMulAutocast(PrecisionMode)"/>). The fused compiled
/// step (<see cref="Training.FusedTrainingSession{T}"/>) and the eager tape steps open the same scope, so both paths
/// cast the same products and give the same results up to FP32 accumulation order.
/// </summary>
/// <remarks>
/// Unlike PyTorch, a product's result is stored in FP32, not FP16/BF16, and so are the pointwise ops and gradients
/// that follow it. An engine without matmul autocast (the CPU engine) keeps the older approximation: the working
/// weights are rounded to the low precision for the step.
/// </remarks>
internal static class MixedPrecisionAutocast
{
    /// <summary>The precision <paramref name="config"/>'s matrix products multiply in, or null when it has none (FP8).</summary>
    internal static PrecisionMode? MatMulPrecision(MixedPrecisionConfig config) => config.PrecisionType switch
    {
        Enums.MixedPrecisionType.FP16 => PrecisionMode.Float16,
        Enums.MixedPrecisionType.BF16 => PrecisionMode.BFloat16,
        _ => null,
    };

    /// <summary>
    /// Opens the matmul autocast scope of <paramref name="context"/> (its precision and
    /// <see cref="MixedPrecisionConfig.MatMulAutocastMode"/>, as the fused step does) on the current engine, or returns
    /// null when the engine's backend has no matmul autocast or the precision has no matmul form.
    /// </summary>
    internal static IDisposable? TryEnter(MixedPrecisionContext context)
        => AiDotNetEngine.Current is DirectGpuTensorEngine gpu
           && gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.IGemmAutocastBackend
           && MatMulPrecision(context.Config) is { } precision
            ? gpu.EnterMatMulAutocast(precision, context.Config.MatMulAutocastMode)
            : null;
}

/// <summary>
/// The working-weight snapshots of an eager mixed-precision step that autocasts its products instead of rounding its
/// weights: there is nothing to restore, and restoring the master weights closes the step's autocast scope, so the
/// optimizer update runs outside it, as PyTorch's does.
/// </summary>
internal sealed class AutocastWorkingWeights : List<float[]>, IDisposable
{
    private IDisposable? _scope;

    internal AutocastWorkingWeights(IDisposable scope) => _scope = scope;

    /// <summary>Closes the autocast scope. Idempotent.</summary>
    public void Dispose() => System.Threading.Interlocked.Exchange(ref _scope, null)?.Dispose();
}
