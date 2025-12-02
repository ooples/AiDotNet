namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents split operation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Splits a tensor into multiple parts along a specified axis.
/// </para>
/// </remarks>
public class SplitOp : IROp
{
    /// <summary>
    /// The axis along which to split.
    /// </summary>
    public int Axis { get; set; }

    /// <summary>
    /// The sizes of each split section.
    /// </summary>
    public int[] SplitSizes { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Number of equal splits (alternative to SplitSizes).
    /// </summary>
    public int NumSplits { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var sizesStr = SplitSizes.Length > 0 ? $"[{string.Join(",", SplitSizes)}]" : $"num={NumSplits}";
        return $"t{OutputId} = Split(t{InputIds[0]}, axis={Axis}, {sizesStr}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
