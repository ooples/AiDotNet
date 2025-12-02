namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused Swish/SiLU activation (x * sigmoid(x)).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A popular activation function.
///
/// Swish(x) = x * sigmoid(x)
///
/// Used in EfficientNet and other modern architectures.
/// Fusing avoids computing sigmoid separately.
/// </para>
/// </remarks>
public class FusedSwishOp : IROp
{
    /// <summary>Validates inputs (single input).</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
