namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for DropoutOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = dropout(x, p, mask)
/// Backward: grad_x = grad_y * mask / (1 - p) (using same mask from forward)
/// </para>
/// </remarks>
public class GradDropoutOp : BackwardOp
{
    /// <summary>Dropout probability.</summary>
    public double Probability { get; set; } = 0.5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and dropout mask
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradDropout[p={Probability}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
