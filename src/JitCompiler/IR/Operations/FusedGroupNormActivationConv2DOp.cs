using AiDotNet.Tensors.Engines;

namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused GroupNorm + Activation + Conv2D in a single operation.
/// This is the complete first half of a DiffusionResBlock:
/// output = Conv2D(activation(GroupNorm(input, gamma, beta)), kernel)
/// </summary>
/// <remarks>
/// <para>
/// Fusing these three operations eliminates TWO intermediate tensors:
/// 1. The GroupNorm output (before activation)
/// 2. The activation output (before conv)
///
/// For a 1280-channel 8x8 tensor, each intermediate is 655KB.
/// With 16 ResBlocks x 2 half-blocks x 50 steps = 3200 fusions,
/// this saves ~4GB of allocation per SD15 Predict call.
/// </para>
/// <para>
/// Inputs: [input, gamma, beta, conv_kernel]
/// Output: conv2d(activation(groupnorm(input, gamma, beta)), conv_kernel)
/// </para>
/// </remarks>
public class FusedGroupNormActivationConv2DOp : IROp
{
    public int NumGroups { get; set; } = 32;
    public double Epsilon { get; set; } = 1e-5;
    public FusedActivationType Activation { get; set; } = FusedActivationType.Swish;
    public int[] Stride { get; set; } = [1, 1];
    public int[] Padding { get; set; } = [0, 0];

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // input, gamma, beta, conv_kernel
        if (InputIds.Length != 4) return false;
        if (NumGroups <= 0) return false;
        return true;
    }
}
