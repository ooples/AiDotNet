namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents slice operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Extracts a contiguous slice from a tensor along specified axes.
/// </para>
/// </remarks>
public class SliceOp : IROp
{
    /// <summary>
    /// Start indices for each axis.
    /// </summary>
    public int[] Starts { get; set; } = Array.Empty<int>();

    /// <summary>
    /// End indices for each axis (exclusive).
    /// </summary>
    public int[] Ends { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Step size for each axis.
    /// </summary>
    public int[] Steps { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Axes to slice on.
    /// </summary>
    public int[] Axes { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = Slice(t{InputIds[0]}, starts=[{string.Join(",", Starts)}], ends=[{string.Join(",", Ends)}]) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
