namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused convolution + batch normalization operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines convolution with batch normalization.
///
/// Batch normalization after convolution is extremely common in CNNs.
/// By fusing them, we can:
/// - Fold BN parameters into conv weights (at inference time)
/// - Skip intermediate tensor storage
/// - Reduce memory bandwidth significantly
///
/// This can be 2-3x faster than separate operations!
/// </para>
/// </remarks>
public class FusedConvBatchNormOp : IROp
{
    /// <summary>
    /// Gets or sets the convolution stride.
    /// </summary>
    public int[] Stride { get; set; } = [1, 1];

    /// <summary>
    /// Gets or sets the convolution padding.
    /// </summary>
    public int[] Padding { get; set; } = [0, 0];

    /// <summary>
    /// Gets or sets the batch norm epsilon value.
    /// </summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets the batch norm momentum.
    /// </summary>
    public double Momentum { get; set; } = 0.1;

    /// <summary>
    /// Validates inputs (input, kernel, gamma, beta, running_mean, running_var).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 6) return false;  // input, kernel, gamma, beta, running_mean, running_var
        return true;
    }
}
