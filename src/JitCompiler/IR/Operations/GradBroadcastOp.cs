namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for BroadcastOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = broadcast(x, target_shape)
/// Backward: grad_x = reduce_sum(grad_y, broadcasted_axes)
/// Sum over axes that were broadcasted.
/// </para>
/// </remarks>
public class GradBroadcastOp : BackwardOp
{
    /// <summary>Original shape before broadcast.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Axes that were broadcasted.</summary>
    public int[] BroadcastedAxes { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradBroadcast[axes={string.Join(",", BroadcastedAxes)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
