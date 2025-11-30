namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents SQRBF (Squared Radial Basis Function) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SQRBF(x) = 1 - x^2 if |x| &lt;= 1, else 0.
/// Compactly supported activation function.
/// </para>
/// </remarks>
public class SQRBFOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
