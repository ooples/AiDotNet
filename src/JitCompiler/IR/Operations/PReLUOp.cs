namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents PReLU (Parametric ReLU) activation in the IR.
/// </summary>
public class PReLUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input + alpha parameter
        if (InputIds.Length != 2) return false;
        return true;
    }
}
