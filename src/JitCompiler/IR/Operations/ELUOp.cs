namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents ELU (Exponential Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ELU(x) = x if x > 0, alpha * (exp(x) - 1) otherwise.
/// Smoother than ReLU for negative values.
/// </para>
/// </remarks>
public class ELUOp : IROp
{
    /// <summary>
    /// The alpha parameter for negative values. Default is 1.0.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ELU(t{InputIds[0]}, alpha={Alpha}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
