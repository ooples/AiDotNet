namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents SELU (Scaled Exponential Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1))).
/// Self-normalizing activation with fixed scale and alpha values.
/// </para>
/// </remarks>
public class SELUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
