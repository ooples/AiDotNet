namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Bent Identity activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes BentIdentity(x) = (sqrt(x^2 + 1) - 1) / 2 + x.
/// Smooth approximation to ReLU with non-zero gradients everywhere.
/// </para>
/// </remarks>
public class BentIdentityOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
