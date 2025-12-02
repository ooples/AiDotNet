namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents sum reduction in the IR.
/// </summary>
public class SumOp : IROp
{
    public int[]? Axes { get; set; }
    public bool KeepDims { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var axesStr = Axes != null ? $"[{string.Join(",", Axes)}]" : "all";
        return $"t{OutputId} = Sum(t{InputIds[0]}, axes={axesStr}, keepDims={KeepDims}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
