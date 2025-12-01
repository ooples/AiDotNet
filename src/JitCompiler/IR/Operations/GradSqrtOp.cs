namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SqrtOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = sqrt(x)
/// Backward: grad_x = grad_y / (2 * sqrt(x)) = grad_y / (2 * y)
/// </para>
/// </remarks>
public class GradSqrtOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSqrt(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
