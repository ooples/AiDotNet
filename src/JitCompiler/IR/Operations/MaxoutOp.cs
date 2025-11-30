namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Maxout activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Maxout(x) = max(x_1, x_2, ..., x_k) over k pieces.
/// Piecewise linear activation that learns its shape.
/// </para>
/// </remarks>
public class MaxoutOp : IROp
{
    /// <summary>
    /// Number of linear pieces. Default is 2.
    /// </summary>
    public int NumPieces { get; set; } = 2;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (NumPieces < 2) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Maxout(t{InputIds[0]}, pieces={NumPieces}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
