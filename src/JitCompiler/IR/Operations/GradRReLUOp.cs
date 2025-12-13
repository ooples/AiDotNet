namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for RReLUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x if x >= 0, alpha * x otherwise (alpha is random during training)
/// Backward: grad_x = grad_y if x >= 0, grad_y * alpha otherwise
/// </para>
/// </remarks>
public class GradRReLUOp : BackwardOp
{
    /// <summary>The random negative slope used during forward pass.</summary>
    public double SampledAlpha { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradRReLU[alpha={SampledAlpha}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
