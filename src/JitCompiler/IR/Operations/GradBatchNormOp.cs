namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for BatchNormOp.
/// </summary>
/// <remarks>
/// <para>
/// Batch normalization has complex gradients involving batch statistics.
/// Computes gradients for input, scale, and bias parameters.
/// </para>
/// </remarks>
public class GradBatchNormOp : BackwardOp
{
    public int InputIndex { get; set; } // 0 = input, 1 = scale, 2 = bias
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradBatchNorm[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
