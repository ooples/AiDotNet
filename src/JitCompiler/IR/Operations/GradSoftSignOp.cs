namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SoftSignOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x / (1 + |x|)
/// Backward: grad_x = grad_y / (1 + |x|)^2
/// </para>
/// </remarks>
public class GradSoftSignOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSoftSign(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
