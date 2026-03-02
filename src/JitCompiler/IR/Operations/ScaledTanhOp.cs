namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Scaled Tanh activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ScaledTanh(x) = (1 - exp(-beta*x)) / (1 + exp(-beta*x)) = tanh(beta*x/2).
/// Tanh with adjustable steepness. Equivalent to standard tanh when beta = 2.
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
