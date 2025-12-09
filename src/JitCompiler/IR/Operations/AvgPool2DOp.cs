namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents 2D average pooling in the IR.
/// </summary>
public class AvgPool2DOp : IROp
{
    public int[] PoolSize { get; set; } = new int[] { 2, 2 };
    public int[] Stride { get; set; } = new int[] { 2, 2 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
