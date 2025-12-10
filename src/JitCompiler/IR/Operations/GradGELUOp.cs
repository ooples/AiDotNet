namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for GELUOp.
/// </summary>
/// <remarks>
/// <para>
/// GELU gradient is computed using the derivative of the GELU function.
/// grad_x = grad_y * (0.5 * (1 + tanh(...)) + 0.5 * x * sech^2(...) * derivative_of_inner)
/// </para>
/// </remarks>
public class GradGELUOp : BackwardOp
{
    /// <summary>Whether approximate GELU was used.</summary>
    public bool Approximate { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradGELU[approx={Approximate}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
