namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for CELUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
/// Backward: grad_x = grad_y if x > 0, grad_y * exp(x/alpha) otherwise
/// </para>
/// </remarks>
public class GradCELUOp : BackwardOp
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
        return $"t{OutputId} = GradCELU[alpha={Alpha}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
