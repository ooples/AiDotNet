namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents mean reduction in the IR.
/// </summary>
public class MeanOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
