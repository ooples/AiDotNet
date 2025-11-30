namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Squash activation in the IR (for Capsule Networks).
/// </summary>
/// <remarks>
/// <para>
/// Computes Squash(x) = (||x||^2 / (1 + ||x||^2)) * (x / ||x||).
/// Used in capsule networks to ensure output vectors have length between 0 and 1.
/// </para>
/// </remarks>
public class SquashOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
