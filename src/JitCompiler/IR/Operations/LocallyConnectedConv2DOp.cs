namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents locally connected 2D convolution in the IR.
/// </summary>
public class LocallyConnectedConv2DOp : IROp
{
    public int[] Stride { get; set; } = new int[] { 1, 1 };
    public int[] Padding { get; set; } = new int[] { 0, 0 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 2) return false;
        return true;
    }
}
