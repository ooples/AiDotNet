namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for DepthwiseConv2DOp.
/// </summary>
public class GradDepthwiseConv2DOp : BackwardOp
{
    /// <summary>Which input: 0 = input, 1 = weight.</summary>
    public int InputIndex { get; set; }

    /// <summary>Stride used in forward.</summary>
    public int[] Stride { get; set; } = new int[] { 1, 1 };

    /// <summary>Padding used in forward.</summary>
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradDepthwiseConv2D[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
