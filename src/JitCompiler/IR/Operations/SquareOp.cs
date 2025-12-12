namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents square operation in the IR.
/// </summary>
public class SquareOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
