namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Group Normalization in the IR.
/// Used in diffusion UNet ResBlocks (every DiffusionResBlock does GroupNorm -> SiLU -> Conv).
/// </summary>
/// <remarks>
/// <para>
/// GroupNorm normalizes across channel groups rather than the batch dimension,
/// making it independent of batch size. This is the standard normalization in
/// diffusion models (DDPM, Stable Diffusion) per Ho et al. 2020 and Rombach et al. 2022.
/// </para>
/// <para>
/// Inputs: [input, gamma, beta] where:
/// - input: [batch, channels, height, width]
/// - gamma: [channels] (scale parameter)
/// - beta: [channels] (shift parameter)
/// </para>
/// </remarks>
public class GroupNormOp : IROp
{
    /// <summary>
    /// Number of groups to divide channels into. Standard: 32 (Stable Diffusion default).
    /// </summary>
    public int NumGroups { get; set; } = 32;

    /// <summary>
    /// Small constant for numerical stability in the division.
    /// </summary>
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, gamma, beta
        if (InputIds.Length != 3) return false;
        if (NumGroups <= 0) return false;
        return true;
    }
}
