namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents mean reduction in the IR.
/// </summary>
public class ReduceMeanOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
