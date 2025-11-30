namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents pixel shuffle (depth-to-space) operation in the IR.
/// </summary>
public class PixelShuffleOp : IROp
{
    public int UpscaleFactor { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (UpscaleFactor <= 0) return false;
        return true;
    }
}
