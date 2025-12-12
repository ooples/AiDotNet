namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Taylor Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Uses Taylor series expansion of exp() for approximate softmax.
/// Can be faster than standard softmax for lower orders.
/// </para>
/// </remarks>
public class TaylorSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute Taylor softmax. Default is -1.
    /// </summary>
    public int Axis { get; set; } = -1;

    /// <summary>
    /// Taylor series expansion order. Default is 2.
    /// </summary>
    public int Order { get; set; } = 2;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Order < 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = TaylorSoftmax(t{InputIds[0]}, axis={Axis}, order={Order}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
