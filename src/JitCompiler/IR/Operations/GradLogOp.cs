namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for LogOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = log(x)
/// Backward: grad_x = grad_y / x
/// </para>
/// </remarks>
public class GradLogOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input (x)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradLog(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
