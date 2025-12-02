namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents LiSHT (Linearly Scaled Hyperbolic Tangent) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LiSHT(x) = x * tanh(x).
/// Similar to Swish but with tanh instead of sigmoid.
/// </para>
/// </remarks>
public class LiSHTOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
