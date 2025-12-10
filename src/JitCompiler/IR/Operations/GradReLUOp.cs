namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ReLUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = max(0, x)
/// Backward: grad_x = grad_y * (x > 0)
/// </para>
/// </remarks>
public class GradReLUOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input (x)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradReLU(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
