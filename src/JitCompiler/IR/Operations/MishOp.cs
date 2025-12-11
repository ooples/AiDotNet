namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Mish activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Mish(x) = x * tanh(softplus(x)).
/// Smooth, non-monotonic activation that often outperforms ReLU.
/// </para>
/// </remarks>
public class MishOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
