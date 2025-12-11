namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents GELU (Gaussian Error Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes GELU(x) = x * Φ(x) where Φ is the standard normal CDF.
/// Used in modern transformers (BERT, GPT).
/// </para>
/// </remarks>
public class GELUOp : IROp
{
    /// <summary>
    /// Whether to use the approximate formula.
    /// </summary>
    public bool Approximate { get; set; } = false;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GELU(t{InputIds[0]}, approx={Approximate}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
