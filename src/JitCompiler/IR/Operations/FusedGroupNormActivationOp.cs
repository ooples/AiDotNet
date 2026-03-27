using AiDotNet.Tensors.Engines;

namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused GroupNorm + Activation in a single pass over the data.
/// Eliminates the intermediate tensor between GroupNorm and activation.
/// </summary>
/// <remarks>
/// <para>
/// The most repeated pattern in diffusion UNets is GroupNorm -> SiLU.
/// Each DiffusionResBlock does this twice (before each Conv3x3).
/// With 16+ ResBlocks in a UNet and 50 denoising steps,
/// this fusion eliminates ~1600 intermediate tensor allocations per Predict call.
/// </para>
/// <para>
/// At 1280 channels on 8x8 spatial (deepest UNet level):
/// each intermediate = 1280 * 8 * 8 * 8 bytes = 655KB.
/// 1600 eliminations = ~1GB less allocation per Predict.
/// </para>
/// <para>
/// Inputs: [input, gamma, beta] (same as GroupNormOp)
/// Output: activation(groupnorm(input, gamma, beta))
/// </para>
/// </remarks>
public class FusedGroupNormActivationOp : IROp
{
    /// <summary>
    /// Number of groups for GroupNorm. Standard: 32.
    /// </summary>
    public int NumGroups { get; set; } = 32;

    /// <summary>
    /// Epsilon for numerical stability.
    /// </summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>
    /// The activation to apply after normalization.
    /// FusedActivationType.Swish for SiLU (the diffusion standard).
    /// </summary>
    public FusedActivationType Activation { get; set; } = FusedActivationType.Swish;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;
        if (NumGroups <= 0) return false;
        return true;
    }
}
