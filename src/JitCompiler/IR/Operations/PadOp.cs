namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents padding operation in the IR.
/// </summary>
public class PadOp : IROp
{
    /// <summary>Padding width per dimension as 2D array [dim, (before, after)].</summary>
    public int[,]? PadWidth { get; set; }

    /// <summary>Simplified padding as 1D array [pad_before_0, pad_after_0, pad_before_1, pad_after_1, ...].</summary>
    public int[] Padding { get; set; } = Array.Empty<int>();

    /// <summary>Input shape for kernel generation.</summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
