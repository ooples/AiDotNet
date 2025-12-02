namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SumOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = sum(x, axes)
/// Backward: grad_x = broadcast(grad_y, original_shape)
/// Gradient is broadcasted back to original shape.
/// </para>
/// </remarks>
public class GradSumOp : BackwardOp
{
    /// <summary>Original input shape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Axes that were reduced.</summary>
    public int[]? Axes { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var axesStr = Axes != null ? string.Join(",", Axes) : "all";
        return $"t{OutputId} = GradSum[axes={axesStr}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
