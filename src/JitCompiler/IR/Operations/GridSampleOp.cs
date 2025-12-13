namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents grid sampling for spatial transformer in the IR.
/// </summary>
public class GridSampleOp : IROp
{
    public string InterpolationMode { get; set; } = "bilinear";
    public string PaddingMode { get; set; } = "zeros";

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;  // input, grid
        return true;
    }
}
