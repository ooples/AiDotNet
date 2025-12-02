namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Scaled Tanh activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ScaledTanh(x) = tanh(beta * x).
/// Tanh with adjustable steepness.
/// </para>
/// </remarks>
public class ScaledTanhOp : IROp
{
    /// <summary>
    /// Scaling factor for input. Default is 1.0.
    /// </summary>
    public double Beta { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ScaledTanh(t{InputIds[0]}, beta={Beta}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
