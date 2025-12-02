namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents CELU (Continuously Differentiable ELU) activation in the IR.
/// </summary>
public class CELUOp : IROp
{
    /// <summary>
    /// The alpha parameter. Default is 1.0.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
