using AiDotNet.Tensors.Engines;

namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused Conv2D + Bias + Activation in a single kernel.
/// Maps directly to IEngine.FusedConv2D for maximum hardware utilization.
/// </summary>
/// <remarks>
/// <para>
/// Different from FusedConvBatchNormActivationOp which includes BatchNorm.
/// The diffusion UNet uses Conv + Bias + Identity (activation is applied separately
/// after GroupNorm), but standalone ConvolutionalLayers with ReLU/SiLU benefit from
/// this fusion directly.
/// </para>
/// <para>
/// Inputs: [input, kernel, bias]
/// Output: activation(conv2d(input, kernel) + bias)
/// </para>
/// </remarks>
public class FusedConv2DBiasActivationOp : IROp
{
    public int[] Stride { get; set; } = [1, 1];
    public int[] Padding { get; set; } = [0, 0];
    public int[] Dilation { get; set; } = [1, 1];
    public FusedActivationType Activation { get; set; } = FusedActivationType.None;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, kernel, bias
        if (InputIds.Length != 3) return false;
        if (Stride[0] <= 0 || Stride[1] <= 0) return false;
        return true;
    }
}
