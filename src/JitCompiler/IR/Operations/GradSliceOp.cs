namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SliceOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = slice(x, start, end)
/// Backward: grad_x = pad_with_zeros(grad_y, original_shape, start_indices)
/// Gradient is zero everywhere except the sliced region.
/// </para>
/// </remarks>
public class GradSliceOp : BackwardOp
{
    /// <summary>Original input shape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    /// <summary>Start indices for the slice.</summary>
    public int[] StartIndices { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSlice[start={string.Join(",", StartIndices)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
