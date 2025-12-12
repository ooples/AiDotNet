namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents layer normalization in the IR.
/// </summary>
public class LayerNormOp : IROp
{
    public int[] NormalizedShape { get; set; } = Array.Empty<int>();
    public double Epsilon { get; set; } = 1e-5;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, gamma, beta
        if (InputIds.Length != 3) return false;
        return true;
    }
}
