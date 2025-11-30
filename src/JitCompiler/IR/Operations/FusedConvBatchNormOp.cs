namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents fused Conv + BatchNorm operation in the IR.
/// </summary>
public class FusedConvBatchNormOp : IROp
{
    /// <summary>
    /// Convolution stride.
    /// </summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>
    /// Convolution padding.
    /// </summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    /// <summary>
    /// BatchNorm epsilon.
    /// </summary>
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: input, conv_weights, bn_gamma, bn_beta, bn_running_mean, bn_running_var
        if (InputIds.Length != 6) return false;
        return true;
    }
}
