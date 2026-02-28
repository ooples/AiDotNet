namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ScaledTanhOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = (1 - exp(-beta*x)) / (1 + exp(-beta*x)) = tanh(beta*x/2)
/// Backward: grad_x = grad_y * (beta/2) * (1 - y^2)
/// </para>
/// </remarks>
public class GradScaledTanhOp : BackwardOp
{
    /// <summary>Beta parameter used in forward.</summary>
    public double Beta { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradScaledTanh[beta={Beta}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
