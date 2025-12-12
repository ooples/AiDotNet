namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SwishOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x * sigmoid(x)
/// Backward: grad_x = grad_y * (y + sigmoid(x) * (1 - y))
/// </para>
/// </remarks>
public class GradSwishOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSwish(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
