namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for BentIdentityOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = (sqrt(x^2 + 1) - 1) / 2 + x
/// Backward: grad_x = grad_y * (x / (2 * sqrt(x^2 + 1)) + 1)
/// </para>
/// </remarks>
public class GradBentIdentityOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradBentIdentity(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
