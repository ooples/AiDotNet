namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for LeakyReLUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = max(alpha * x, x)
/// Backward: grad_x = grad_y * (1 if x > 0 else alpha)
/// </para>
/// </remarks>
public class GradLeakyReLUOp : BackwardOp
{
    /// <summary>Negative slope.</summary>
    public double Alpha { get; set; } = 0.01;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradLeakyReLU[alpha={Alpha}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
