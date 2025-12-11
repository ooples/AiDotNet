namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for Conv2DOp.
/// </summary>
/// <remarks>
/// <para>
/// Computes gradient for convolution inputs (data, filters, or bias).
/// Uses convolution theorems for efficient gradient computation.
/// </para>
/// </remarks>
public class GradConv2DOp : BackwardOp
{
    public int InputIndex { get; set; } // 0 = data, 1 = filters, 2 = bias
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs depend on which gradient we're computing
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradConv2D[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
