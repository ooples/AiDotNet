namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for LiSHTOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x * tanh(x)
/// Backward: grad_x = grad_y * (tanh(x) + x * sech^2(x))
/// </para>
/// </remarks>
public class GradLiSHTOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradLiSHT(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
