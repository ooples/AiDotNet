namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SELUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
/// Backward: grad_x = grad_y * scale if x > 0, grad_y * scale * alpha * exp(x) otherwise
/// </para>
/// </remarks>
public class GradSELUOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSELU(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
