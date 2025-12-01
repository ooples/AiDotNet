namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ISRUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x / sqrt(1 + alpha * x^2)
/// Backward: grad_x = grad_y / (1 + alpha * x^2)^(3/2)
/// </para>
/// </remarks>
public class GradISRUOp : BackwardOp
{
    /// <summary>Alpha parameter used in forward.</summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradISRU[alpha={Alpha}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
