namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for HardSigmoidOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = clip((x + 3) / 6, 0, 1)
/// Backward: grad_x = grad_y / 6 if -3 &lt; x &lt; 3, else 0
/// </para>
/// </remarks>
public class GradHardSigmoidOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradHardSigmoid(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
