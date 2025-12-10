namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused residual block operation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Fuses a residual/skip connection pattern.
///
/// Residual blocks are everywhere in modern networks (ResNet, Transformers, etc.)
/// Pattern:
///   output = activation(main_path + skip_connection)
///
/// By fusing this, we can:
/// - Optimize the addition and activation together
/// - Reduce memory traffic
/// - Better utilize CPU/GPU resources
/// </para>
/// </remarks>
public class FusedResidualBlockOp : IROp
{
    /// <summary>
    /// Gets or sets the activation function name.
    /// </summary>
    public string ActivationName { get; set; } = "ReLU";

    /// <summary>
    /// Validates inputs (main_path, skip_connection).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        if (string.IsNullOrEmpty(ActivationName)) return false;
        return true;
    }
}
