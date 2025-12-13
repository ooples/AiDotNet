namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents SoftSign activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SoftSign(x) = x / (1 + |x|).
/// Alternative to tanh with polynomial tails.
/// </para>
/// </remarks>
public class SoftSignOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
