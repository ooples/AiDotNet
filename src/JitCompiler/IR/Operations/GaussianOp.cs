namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Gaussian activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Gaussian(x) = exp(-x^2).
/// Bell-shaped activation centered at zero.
/// </para>
/// </remarks>
public class GaussianOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
