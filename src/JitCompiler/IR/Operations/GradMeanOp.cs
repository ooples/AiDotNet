namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for MeanOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = mean(x, axes)
/// Backward: grad_x = broadcast(grad_y / count, original_shape)
/// Similar to sum but divided by number of elements.
/// </para>
/// </remarks>
public class GradMeanOp : BackwardOp
{
    /// <summary>Original input shape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Axes that were reduced.</summary>
    public int[]? Axes { get; set; }

    /// <summary>Number of elements that were averaged.</summary>
    public int Count { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMean[count={Count}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
