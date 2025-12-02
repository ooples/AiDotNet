namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused batch normalization + activation operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines batch norm with activation.
///
/// BatchNorm followed by ReLU is extremely common in CNNs.
/// Fusing them reduces memory traffic and improves performance.
///
/// Pattern:
///   x_norm = (x - mean) / sqrt(var + epsilon)
///   output = activation(gamma * x_norm + beta)
/// </para>
/// </remarks>
public class FusedBatchNormActivationOp : IROp
{
    /// <summary>Gets or sets the activation function name.</summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>Gets or sets epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Gets or sets momentum for running statistics.</summary>
    public double Momentum { get; set; } = 0.1;

    /// <summary>Validates inputs.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 5) return false; // input, gamma, beta, running_mean, running_var
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
