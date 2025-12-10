namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for LayerNormOp.
/// </summary>
/// <remarks>
/// <para>
/// Layer normalization gradient is complex, involving variance and mean.
/// Computes gradients for input, gamma, and beta.
/// </para>
/// </remarks>
public class GradLayerNormOp : BackwardOp
{
    /// <summary>Which input: 0 = input, 1 = gamma, 2 = beta.</summary>
    public int InputIndex { get; set; }

    /// <summary>Epsilon for numerical stability.</summary>
    public double Epsilon { get; set; } = 1e-5;

    /// <summary>Normalized shape.</summary>
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradLayerNorm[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
