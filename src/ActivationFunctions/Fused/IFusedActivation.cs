using AiDotNet.Tensors.Engines;

namespace AiDotNet.ActivationFunctions.Fused;

/// <summary>
/// Implemented by scalar activation functions that have an exact fused-kernel
/// equivalent (<see cref="FusedActivationType"/>), so fused inference paths such
/// as <c>IEngine.MlpForward</c> can ask the activation what kernel it maps to
/// instead of switching on its type.
/// </summary>
/// <remarks>
/// <para>
/// Open/closed-compliant for the same reason as
/// <see cref="Optimizers.Fused.IFusedOptimizerSpec"/>: only activations whose
/// fused kernel is numerically identical to the scalar form implement this, so
/// there is no central activation→enum switch to maintain and an unrecognized
/// activation simply keeps the generic per-layer path. A null activation (linear
/// layer) is treated as <see cref="FusedActivationType.None"/> by callers.
/// </para>
/// <para>
/// <see cref="TryGetFusedActivation"/> returns <c>false</c> when THIS instance
/// can't be reproduced by the fused kernel — e.g. a parametric activation whose
/// parameter (LeakyReLU slope, ELU alpha) differs from the value the kernel
/// hardcodes — so a custom-parameter instance correctly falls back rather than
/// silently getting the kernel's default parameter.
/// </para>
/// </remarks>
public interface IFusedActivation
{
    /// <summary>
    /// Reports the fused-kernel activation type equivalent to this activation, or
    /// returns <c>false</c> if this instance can't be reproduced by the kernel.
    /// </summary>
    bool TryGetFusedActivation(out FusedActivationType type);
}
