namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SoftmaxOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y_i = exp(x_i) / sum(exp(x_j))
/// Backward: grad_x = y * (grad_y - sum(grad_y * y))
/// (Jacobian computation for softmax)
/// </para>
/// </remarks>
public class GradSoftmaxOp : BackwardOp
{
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSoftmax[axis={Axis}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
