using AiDotNet.Tensors.Engines;

namespace AiDotNet.ActivationFunctions.Fused;

/// <summary>
/// Implemented by scalar activation functions that have an exact fused-kernel
/// equivalent (<see cref="FusedActivationType"/>), so fused inference paths such
/// as <c>IEngine.MlpForward</c> can ask the activation what kernel it maps to
/// instead of switching on its type.
/// </summary>
/// <remarks>
/// Open/closed-compliant for the same reason as
/// <see cref="Optimizers.Fused.IFusedOptimizerSpec"/>: only activations whose
/// fused kernel is numerically identical to the scalar form implement this, so
/// there is no central activation→enum switch to maintain and an unrecognized
/// activation simply keeps the generic per-layer path. A null activation (linear
/// layer) is treated as <see cref="FusedActivationType.None"/> by callers.
/// </remarks>
public interface IFusedActivation
{
    /// <summary>The fused-kernel activation type equivalent to this activation.</summary>
    FusedActivationType FusedActivationType { get; }
}
