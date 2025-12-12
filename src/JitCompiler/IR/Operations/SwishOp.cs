namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Swish/SiLU activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Swish(x) = x * sigmoid(x).
/// Self-gated activation with smooth gradient.
/// </para>
/// </remarks>
public class SwishOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
