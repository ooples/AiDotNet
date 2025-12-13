namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ExpOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = exp(x)
/// Backward: grad_x = grad_y * y
/// (derivative of exp is exp itself)
/// </para>
/// </remarks>
public class GradExpOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradExp(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
