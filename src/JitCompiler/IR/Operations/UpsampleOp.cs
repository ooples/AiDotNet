namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents upsampling operation in the IR.
/// </summary>
public class UpsampleOp : IROp
{
    /// <summary>Upsampling scale factor.</summary>
    public int Scale { get; set; } = 2;

    /// <summary>Upsampling mode: "nearest" or "bilinear".</summary>
    public string Mode { get; set; } = "nearest";

    /// <summary>Input shape [batch, channels, height, width] for kernel generation.</summary>
    public int[] InputShape { get; set; } = new int[] { 1, 1, 1, 1 };

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Scale <= 0) return false;
        return true;
    }
}
