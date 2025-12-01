namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused Conv2D + BatchNorm + Activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The complete CNN layer in one operation!
///
/// Combines:
///   1. Convolution
///   2. Batch normalization
///   3. Activation (ReLU, etc.)
///
/// This is THE most common pattern in CNNs.
/// Can be 3-5x faster than separate operations.
/// </para>
/// </remarks>
public class FusedConvBatchNormActivationOp : IROp
{
    /// <summary>Gets or sets the convolution stride.</summary>
    public int[] Stride { get; set; } = [1, 1];

    /// <summary>Gets or sets the convolution padding.</summary>
    public int[] Padding { get; set; } = [0, 0];

    /// <summary>Gets or sets the batch norm epsilon.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Gets or sets the batch norm momentum.</summary>
    public double Momentum { get; set; } = 0.1;

    /// <summary>Gets or sets the activation function name.</summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>Validates inputs.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 6) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
