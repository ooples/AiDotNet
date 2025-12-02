namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ElementwiseMultiplyOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a * b (element-wise)
/// Backward: grad_a = grad_c * b, grad_b = grad_c * a
/// </para>
/// </remarks>
public class GradElementwiseMultiplyOp : BackwardOp
{
    /// <summary>
    /// Which input are we computing the gradient for? (0 = left, 1 = right)
    /// </summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and the other input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradElemMul[input={InputIndex}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
