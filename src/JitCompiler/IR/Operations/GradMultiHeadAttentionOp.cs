namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for multi-head attention.
/// </summary>
public class GradMultiHeadAttentionOp : BackwardOp
{
    /// <summary>Number of attention heads.</summary>
    public int NumHeads { get; set; } = 8;

    /// <summary>Dimension per head.</summary>
    public int HeadDim { get; set; } = 64;

    /// <summary>Which input: 0 = query, 1 = key, 2 = value, 3 = output_projection.</summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMHA[heads={NumHeads}, dim={HeadDim}, input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
