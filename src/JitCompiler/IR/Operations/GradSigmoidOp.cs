namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SigmoidOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = 1 / (1 + exp(-x))
/// Backward: grad_x = grad_y * y * (1 - y)
/// </para>
/// </remarks>
public class GradSigmoidOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output (y)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSigmoid(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
