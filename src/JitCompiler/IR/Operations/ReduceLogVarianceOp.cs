namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents log variance reduction in the IR.
/// </summary>
public class ReduceLogVarianceOp : IROp
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
