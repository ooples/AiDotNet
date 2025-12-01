namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Hard Sigmoid activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes HardSigmoid(x) = clip((x + 3) / 6, 0, 1).
/// Faster piecewise linear approximation of sigmoid.
/// </para>
/// </remarks>
public class HardSigmoidOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
