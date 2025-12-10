namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for TanhOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = tanh(x)
/// Backward: grad_x = grad_y * (1 - y^2)
/// </para>
/// </remarks>
public class GradTanhOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradTanh(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
