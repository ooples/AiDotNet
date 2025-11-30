namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SoftPlusOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = ln(1 + exp(x))
/// Backward: grad_x = grad_y * sigmoid(x)
/// </para>
/// </remarks>
public class GradSoftPlusOp : BackwardOp
{
    /// <summary>Scaling factor used in forward.</summary>
    public double Beta { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSoftPlus[beta={Beta}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
