namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ELUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x if x > 0, alpha * (exp(x) - 1) otherwise
/// Backward: grad_x = grad_y if x > 0, grad_y * alpha * exp(x) otherwise
/// </para>
/// </remarks>
public class GradELUOp : BackwardOp
{
    /// <summary>The alpha parameter.</summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradELU[alpha={Alpha}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
