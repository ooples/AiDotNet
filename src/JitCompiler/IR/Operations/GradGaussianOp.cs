namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for GaussianOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = exp(-x^2)
/// Backward: grad_x = grad_y * (-2 * x * exp(-x^2)) = -2 * x * y * grad_y
/// </para>
/// </remarks>
public class GradGaussianOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradGaussian(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
